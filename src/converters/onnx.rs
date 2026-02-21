use crate::converters::{ConvertedGraph, operand_name};
use crate::debug_print;
use crate::error::GraphError;
use crate::graph::{
    DataType, Dimension, GraphInfo, OperandKind, Operation, get_static_or_max_size,
};
use crate::protos::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, attribute_proto::AttributeType,
    tensor_proto::DataType as ProtoDataType, type_proto::Tensor as TensorTypeProto,
};
use crate::shape_inference::{broadcast_shapes, infer_matmul_shape, infer_transpose_shape};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use prost::Message;
use webnn_onnx_utils::{
    attributes::AttrBuilder, data_types as utils_data_types, operation_names::mapper,
    tensor_data::TensorData,
};

#[derive(Default)]
pub struct OnnxConverter;

impl OnnxConverter {
    fn parse_f64_attr(value: Option<&serde_json::Value>) -> Option<f64> {
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

    fn parse_dynamic_dim_expr(dim_name: &str) -> (String, i64) {
        let s = dim_name.trim();
        if let Some((lhs, rhs)) = s.rsplit_once('+')
            && let Ok(offset) = rhs.trim().parse::<i64>()
        {
            return (lhs.trim().to_string(), offset);
        }
        if let Some((lhs, rhs)) = s.rsplit_once('-')
            && let Ok(offset) = rhs.trim().parse::<i64>()
        {
            return (lhs.trim().to_string(), -offset);
        }
        (s.to_string(), 0)
    }

    fn parse_dimension_array(value: &serde_json::Value) -> Option<Vec<Dimension>> {
        let arr = value.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            if let Some(n) = v.as_u64() {
                out.push(Dimension::Static(n as u32));
                continue;
            }
            if let Some(n) = v.as_i64() {
                if n < 0 {
                    return None;
                }
                out.push(Dimension::Static(n as u32));
                continue;
            }
            let obj = v.as_object()?;
            let name = obj.get("name")?.as_str()?.to_string();
            let max_size = obj
                .get("maxSize")
                .or_else(|| obj.get("max_size"))?
                .as_u64()? as u32;
            out.push(Dimension::Dynamic(crate::graph::DynamicDimension {
                name,
                max_size,
            }));
        }
        Some(out)
    }

    fn resolve_dynamic_dim_source(
        graph: &GraphInfo,
        op: &Operation,
        dim_name: &str,
    ) -> Option<(String, usize)> {
        if dim_name.is_empty() {
            return None;
        }

        // Prefer op inputs, then graph inputs for runtime-available sources.
        for id in op
            .input_operands
            .iter()
            .copied()
            .chain(graph.input_operands.iter().copied())
        {
            let operand = graph.operand(id)?;
            for (axis, dim) in operand.descriptor.shape.iter().enumerate() {
                if let Dimension::Dynamic(dd) = dim
                    && dd.name == dim_name
                {
                    return Some((operand_name(graph, id), axis));
                }
            }
        }
        None
    }

    fn build_runtime_shape_input(
        prefix: &str,
        target_shape: &[Dimension],
        graph: &GraphInfo,
        op: &Operation,
        nodes: &mut Vec<NodeProto>,
        initializers: &mut Vec<TensorProto>,
    ) -> String {
        let mut parts = Vec::with_capacity(target_shape.len());
        let mut shape_cache: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        for (idx, dim) in target_shape.iter().enumerate() {
            match dim {
                Dimension::Static(v) => {
                    let name = format!("{}_dim{}_const", prefix, idx);
                    initializers.push(TensorProto {
                        name: name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![1],
                        int64_data: vec![*v as i64],
                        ..Default::default()
                    });
                    parts.push(name);
                }
                Dimension::Dynamic(dd) => {
                    let (base_name, offset) = Self::parse_dynamic_dim_expr(&dd.name);
                    if let Some((source_tensor, axis)) =
                        Self::resolve_dynamic_dim_source(graph, op, &base_name)
                    {
                        let shape_output = if let Some(existing) = shape_cache.get(&source_tensor) {
                            existing.clone()
                        } else {
                            let shape_name = format!("{}_{}_shape", prefix, source_tensor);
                            nodes.push(NodeProto {
                                input: vec![source_tensor.clone()],
                                output: vec![shape_name.clone()],
                                name: format!("{}_{}_shape_node", prefix, source_tensor),
                                op_type: "Shape".to_string(),
                                attribute: vec![],
                                ..Default::default()
                            });
                            shape_cache.insert(source_tensor.clone(), shape_name.clone());
                            shape_name
                        };

                        let index_name = format!("{}_dim{}_index", prefix, idx);
                        initializers.push(TensorProto {
                            name: index_name.clone(),
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![1],
                            int64_data: vec![axis as i64],
                            ..Default::default()
                        });

                        let gather_out = format!("{}_dim{}_value", prefix, idx);
                        nodes.push(NodeProto {
                            input: vec![shape_output, index_name],
                            output: vec![gather_out.clone()],
                            name: format!("{}_dim{}_gather", prefix, idx),
                            op_type: "Gather".to_string(),
                            attribute: vec![],
                            ..Default::default()
                        });
                        if offset == 0 {
                            parts.push(gather_out);
                        } else {
                            let offset_name = format!("{}_dim{}_offset", prefix, idx);
                            initializers.push(TensorProto {
                                name: offset_name.clone(),
                                data_type: ProtoDataType::Int64 as i32,
                                dims: vec![1],
                                int64_data: vec![offset],
                                ..Default::default()
                            });
                            let shifted_out = format!("{}_dim{}_value_shifted", prefix, idx);
                            nodes.push(NodeProto {
                                input: vec![gather_out, offset_name],
                                output: vec![shifted_out.clone()],
                                name: format!("{}_dim{}_add_offset", prefix, idx),
                                op_type: "Add".to_string(),
                                attribute: vec![],
                                ..Default::default()
                            });
                            parts.push(shifted_out);
                        }
                    } else {
                        debug_print!(
                            "[ONNX CONVERTER] Could not resolve dynamic dim '{}' (base='{}') for {}; falling back to max_size={}",
                            dd.name,
                            base_name,
                            prefix,
                            dd.max_size
                        );
                        let name = format!("{}_dim{}_fallback", prefix, idx);
                        initializers.push(TensorProto {
                            name: name.clone(),
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![1],
                            int64_data: vec![dd.max_size as i64],
                            ..Default::default()
                        });
                        parts.push(name);
                    }
                }
            }
        }

        if parts.len() == 1 {
            return parts[0].clone();
        }

        let shape_name = format!("{}_runtime_shape", prefix);
        nodes.push(NodeProto {
            input: parts,
            output: vec![shape_name.clone()],
            name: format!("{}_runtime_shape_concat", prefix),
            op_type: "Concat".to_string(),
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                r#type: AttributeType::Int as i32,
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        });
        shape_name
    }

    fn create_runtime_filled_tensor(
        prefix: &str,
        fill_value: f32,
        dtype: ProtoDataType,
        shape_tensor_name: String,
        nodes: &mut Vec<NodeProto>,
        initializers: &mut Vec<TensorProto>,
    ) -> String {
        let scalar_name = format!("{}_scalar", prefix);
        initializers.push(Self::create_scalar_initializer(
            scalar_name.clone(),
            dtype,
            fill_value,
        ));

        let output_name = format!("{}_filled", prefix);
        nodes.push(NodeProto {
            input: vec![scalar_name, shape_tensor_name],
            output: vec![output_name.clone()],
            name: format!("{}_expand", prefix),
            op_type: "Expand".to_string(),
            attribute: vec![],
            ..Default::default()
        });
        output_name
    }

    fn build_norm_layer_shape_vector(
        prefix: &str,
        input_name: String,
        axis: usize,
        rank: usize,
        nodes: &mut Vec<NodeProto>,
        initializers: &mut Vec<TensorProto>,
    ) -> String {
        let shape_name = format!("{}_shape", prefix);
        nodes.push(NodeProto {
            input: vec![input_name],
            output: vec![shape_name.clone()],
            name: format!("{}_shape_node", prefix),
            op_type: "Shape".to_string(),
            attribute: vec![],
            ..Default::default()
        });

        let starts_name = format!("{}_starts", prefix);
        let ends_name = format!("{}_ends", prefix);
        let axes_name = format!("{}_axes", prefix);
        let steps_name = format!("{}_steps", prefix);

        initializers.push(TensorProto {
            name: starts_name.clone(),
            data_type: ProtoDataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![axis as i64],
            ..Default::default()
        });
        initializers.push(TensorProto {
            name: ends_name.clone(),
            data_type: ProtoDataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![rank as i64],
            ..Default::default()
        });
        initializers.push(TensorProto {
            name: axes_name.clone(),
            data_type: ProtoDataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![0],
            ..Default::default()
        });
        initializers.push(TensorProto {
            name: steps_name.clone(),
            data_type: ProtoDataType::Int64 as i32,
            dims: vec![1],
            int64_data: vec![1],
            ..Default::default()
        });

        let out_name = format!("{}_layer_shape", prefix);
        nodes.push(NodeProto {
            input: vec![shape_name, starts_name, ends_name, axes_name, steps_name],
            output: vec![out_name.clone()],
            name: format!("{}_slice_shape", prefix),
            op_type: "Slice".to_string(),
            attribute: vec![],
            ..Default::default()
        });
        out_name
    }

    fn invalid_operand(
        context: &str,
        operand: u32,
        op_info: Option<(&Operation, usize)>,
    ) -> GraphError {
        if let Some((op, idx)) = op_info {
            debug_print!(
                "[DEBUG] Invalid operand {} at {} (op #{} type={} label={:?} inputs={:?} outputs={:?})",
                operand,
                context,
                idx,
                op.op_type,
                op.label,
                op.input_operands,
                op.output_operands
            );
        } else {
            debug_print!("[DEBUG] Invalid operand {} at {}", operand, context);
        }
        GraphError::InvalidConversionOperand { operand }
    }

    fn data_type_code(data_type: DataType) -> ProtoDataType {
        // Convert rust-webnn-graph DataType to webnn_onnx_utils DataType first
        let utils_dtype = match data_type {
            // ORT does not accept native int4/uint4 tensor inputs in our current path.
            // Keep uint4 as uint8 and int4 as int32 to match runtime input marshaling.
            DataType::Int4 => utils_data_types::DataType::Int32,
            DataType::Uint4 => utils_data_types::DataType::Uint8,
            DataType::Float32 => utils_data_types::DataType::Float32,
            DataType::Float16 => utils_data_types::DataType::Float16,
            DataType::Int32 => utils_data_types::DataType::Int32,
            DataType::Uint32 => utils_data_types::DataType::Uint32,
            DataType::Int64 => utils_data_types::DataType::Int64,
            DataType::Uint64 => utils_data_types::DataType::Uint64,
            DataType::Int8 => utils_data_types::DataType::Int8,
            DataType::Uint8 => utils_data_types::DataType::Uint8,
        };
        // Use shared library conversion
        utils_data_types::webnn_to_onnx(utils_dtype)
    }

    fn create_scalar_initializer(name: String, dtype: ProtoDataType, value: f32) -> TensorProto {
        // Convert ProtoDataType to utils DataType
        let utils_dtype = utils_data_types::onnx_proto_to_webnn(dtype)
            .unwrap_or(utils_data_types::DataType::Float32);

        // Use shared library to create scalar tensor
        TensorData::scalar(utils_dtype.clone(), value).to_tensor_proto(name, utils_dtype, vec![])
    }

    /// Create a vector initializer with proper data type handling
    fn create_vector_initializer(
        name: String,
        dtype: ProtoDataType,
        shape: Vec<i64>,
        value: f32,
    ) -> TensorProto {
        // Convert ProtoDataType to utils DataType
        let utils_dtype = utils_data_types::onnx_proto_to_webnn(dtype)
            .unwrap_or(utils_data_types::DataType::Float32);

        // Use shared library to create filled tensor
        TensorData::filled(utils_dtype.clone(), &shape, value).to_tensor_proto(
            name,
            utils_dtype,
            shape,
        )
    }

    /// Create a Reshape node and accompanying shape initializer. Returns output name.
    fn create_reshape_node(
        prefix: &str,
        input: String,
        target_shape: Vec<i64>,
        nodes: &mut Vec<NodeProto>,
        initializers: &mut Vec<TensorProto>,
    ) -> String {
        let shape_const_name = format!("{}_shape", prefix);
        initializers.push(TensorProto {
            name: shape_const_name.clone(),
            data_type: ProtoDataType::Int64 as i32,
            dims: vec![target_shape.len() as i64],
            int64_data: target_shape.clone(),
            ..Default::default()
        });

        let output_name = format!("{}_reshaped", prefix);
        nodes.push(NodeProto {
            input: vec![input, shape_const_name],
            output: vec![output_name.clone()],
            name: format!("{}_reshape", prefix),
            op_type: "Reshape".to_string(),
            attribute: vec![],
            ..Default::default()
        });
        output_name
    }

    fn onnx_op_type(op_type: &str) -> String {
        let normalized = op_type
            .chars()
            .filter(|c| *c != '_' && *c != '-')
            .flat_map(|c| c.to_lowercase())
            .collect::<String>();
        if normalized == "cumulativesum" {
            return "CumSum".to_string();
        }
        if normalized == "roundeven" {
            return "Round".to_string();
        }
        if normalized == "grucell" {
            return "GRU".to_string();
        }
        if normalized == "argmax" {
            return "ArgMax".to_string();
        }
        if normalized == "argmin" {
            return "ArgMin".to_string();
        }
        if normalized == "averagepool2d" {
            return "AveragePool".to_string();
        }
        if normalized == "maxpool2d" {
            return "MaxPool".to_string();
        }
        if normalized == "batchnormalization" {
            return "BatchNormalization".to_string();
        }
        if normalized == "instancenormalization" {
            return "InstanceNormalization".to_string();
        }
        if normalized == "layernormalization" {
            return "LayerNormalization".to_string();
        }
        if normalized == "convtranspose2d" {
            return "ConvTranspose".to_string();
        }
        if normalized == "gatherelements" {
            return "GatherElements".to_string();
        }
        if normalized == "gathernd" {
            return "GatherND".to_string();
        }
        if normalized == "quantizelinear" {
            return "QuantizeLinear".to_string();
        }
        if normalized == "dequantizelinear" {
            return "DequantizeLinear".to_string();
        }

        // Use shared operation name mapper from webnn-onnx-utils
        if let Some(onnx_name) = mapper().webnn_to_onnx(op_type) {
            return onnx_name.to_string();
        }

        // Fallback: capitalize first letter for unmapped operations
        let mut chars = op_type.chars();
        if let Some(first) = chars.next() {
            let mut s = first.to_ascii_uppercase().to_string();
            s.push_str(&chars.collect::<String>());
            s
        } else {
            String::new()
        }
    }

    /// Helper: Parse a JSON array attribute as Vec<i64>
    fn parse_i64_array(op: &Operation, json_key: &str) -> Option<Vec<i64>> {
        // Use shared library's parse_json_ints
        webnn_onnx_utils::attributes::parse_json_ints(&op.attributes, json_key)
    }

    fn parse_i64_array_any(op: &Operation, keys: &[&str]) -> Option<Vec<i64>> {
        for key in keys {
            if let Some(values) = Self::parse_i64_array(op, key) {
                return Some(values);
            }
        }
        None
    }

    fn parse_pads_attr(op: &Operation) -> Option<Vec<i64>> {
        if let Some(pads) = Self::parse_i64_array_any(op, &["pads", "paddings"]) {
            return Some(pads);
        }
        let padding = Self::parse_i64_array(op, "padding")?;
        if padding.len() == 4 {
            // WebNN padding is [top, bottom, left, right]
            // ONNX pads is [top, left, bottom, right]
            return Some(vec![padding[0], padding[2], padding[1], padding[3]]);
        }
        Some(padding)
    }

    fn infer_pool_ceil_mode_from_output_sizes(op: &Operation, graph: &GraphInfo) -> Option<i64> {
        let target = Self::parse_i64_array_any(op, &["outputSizes", "output_sizes"])?;
        if target.len() != 2 {
            return None;
        }

        let input_id = *op.input_operands.first()?;
        let input_shape = &graph.operand(input_id)?.descriptor.shape;
        if input_shape.len() != 4 {
            return None;
        }
        let layout = op
            .attributes
            .get("layout")
            .and_then(|v| v.as_str())
            .unwrap_or("nchw")
            .to_ascii_lowercase();
        let (input_h, input_w) = if layout == "nhwc" {
            (
                get_static_or_max_size(&input_shape[1]) as i64,
                get_static_or_max_size(&input_shape[2]) as i64,
            )
        } else {
            (
                get_static_or_max_size(&input_shape[2]) as i64,
                get_static_or_max_size(&input_shape[3]) as i64,
            )
        };

        let kernel = Self::parse_i64_array_any(op, &["windowDimensions", "window_dimensions"]) // WebNN
            .or_else(|| {
                if layout == "nhwc" {
                    Some(vec![
                        get_static_or_max_size(&input_shape[1]) as i64,
                        get_static_or_max_size(&input_shape[2]) as i64,
                    ])
                } else {
                    Some(vec![
                        get_static_or_max_size(&input_shape[2]) as i64,
                        get_static_or_max_size(&input_shape[3]) as i64,
                    ])
                }
            })?;
        if kernel.len() != 2 {
            return None;
        }
        let strides = Self::parse_i64_array(op, "strides").unwrap_or_else(|| vec![1, 1]);
        let dilations = Self::parse_i64_array(op, "dilations").unwrap_or_else(|| vec![1, 1]);
        let pads = Self::parse_pads_attr(op).unwrap_or_else(|| vec![0, 0, 0, 0]);
        if strides.len() != 2 || dilations.len() != 2 || pads.len() != 4 {
            return None;
        }

        let eff_h = dilations[0] * (kernel[0] - 1) + 1;
        let eff_w = dilations[1] * (kernel[1] - 1) + 1;
        let numer_h = input_h + pads[0] + pads[2] - eff_h;
        let numer_w = input_w + pads[1] + pads[3] - eff_w;
        if numer_h < 0 || numer_w < 0 {
            return None;
        }
        let floor_h = (numer_h / strides[0]) + 1;
        let floor_w = (numer_w / strides[1]) + 1;
        let ceil_h = ((numer_h + strides[0] - 1) / strides[0]) + 1;
        let ceil_w = ((numer_w + strides[1] - 1) / strides[1]) + 1;

        if target[0] == floor_h && target[1] == floor_w {
            Some(0)
        } else if target[0] == ceil_h && target[1] == ceil_w {
            Some(1)
        } else {
            None
        }
    }

    fn create_pool2d_attributes_with_graph(
        op: &Operation,
        graph: &GraphInfo,
    ) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        let mut kernel_shape = Self::parse_i64_array_any(
            op,
            &[
                "windowDimensions",
                "window_dimensions",
                "windowdimensions",
                "kernelShape",
                "kernel_shape",
            ],
        )
        .or_else(|| {
            let input_id = *op.input_operands.first()?;
            let input_shape = &graph.operand(input_id)?.descriptor.shape;
            if input_shape.len() != 4 {
                return None;
            }
            let layout = op
                .attributes
                .get("layout")
                .and_then(|v| v.as_str())
                .unwrap_or("nchw")
                .to_ascii_lowercase();
            let (h, w) = if layout == "nhwc" {
                (
                    get_static_or_max_size(&input_shape[1]) as i64,
                    get_static_or_max_size(&input_shape[2]) as i64,
                )
            } else {
                (
                    get_static_or_max_size(&input_shape[2]) as i64,
                    get_static_or_max_size(&input_shape[3]) as i64,
                )
            };
            Some(vec![h, w])
        });
        // WebNN averagePool2d defaults windowDimensions to full spatial extent.
        // Some front-end paths materialize [1,1] with dilations set, which yields
        // incorrect no-op pooling. If output spatial dims indicate reduction, repair
        // to full input window for averagePool.
        let has_explicit_window = op.attributes.get("windowDimensions").is_some()
            || op.attributes.get("window_dimensions").is_some()
            || op.attributes.get("windowdimensions").is_some()
            || op.attributes.get("kernelShape").is_some()
            || op.attributes.get("kernel_shape").is_some();
        if op.op_type.eq_ignore_ascii_case("averagepool2d")
            && op.attributes.get("dilations").is_some()
            && !has_explicit_window
            && let (Some(&in_id), Some(out_id)) = (op.input_operands.first(), op.output_operand)
            && let (Some(in_operand), Some(out_operand)) =
                (graph.operand(in_id), graph.operand(out_id))
            && in_operand.descriptor.shape.len() == 4
            && out_operand.descriptor.shape.len() == 4
        {
            let layout = op
                .attributes
                .get("layout")
                .and_then(|v| v.as_str())
                .unwrap_or("nchw")
                .to_ascii_lowercase();
            let (in_h, in_w, out_h, out_w) = if layout == "nhwc" {
                (
                    get_static_or_max_size(&in_operand.descriptor.shape[1]),
                    get_static_or_max_size(&in_operand.descriptor.shape[2]),
                    get_static_or_max_size(&out_operand.descriptor.shape[1]),
                    get_static_or_max_size(&out_operand.descriptor.shape[2]),
                )
            } else {
                (
                    get_static_or_max_size(&in_operand.descriptor.shape[2]),
                    get_static_or_max_size(&in_operand.descriptor.shape[3]),
                    get_static_or_max_size(&out_operand.descriptor.shape[2]),
                    get_static_or_max_size(&out_operand.descriptor.shape[3]),
                )
            };
            let _ = (out_h, out_w); // output hints may be unreliable for dynamic shapes
            kernel_shape = Some(vec![in_h as i64, in_w as i64]);
        }

        if let Some(kernel_shape) = kernel_shape {
            Self::add_ints_attribute(&mut attributes, "kernel_shape", kernel_shape);
        }
        if let Some(strides) = Self::parse_i64_array(op, "strides") {
            Self::add_ints_attribute(&mut attributes, "strides", strides);
        }
        // ONNX AveragePool does not accept dilations (ORT rejects it).
        let is_max_pool = op.op_type.eq_ignore_ascii_case("maxpool2d");
        if is_max_pool && let Some(dilations) = Self::parse_i64_array(op, "dilations") {
            Self::add_ints_attribute(&mut attributes, "dilations", dilations);
        }
        if let Some(pads) = Self::parse_pads_attr(op) {
            Self::add_ints_attribute(&mut attributes, "pads", pads);
        }
        if let Some(ceil_mode) = if op.attributes.get("outputSizes").is_some()
            || op.attributes.get("output_sizes").is_some()
        {
            Self::infer_pool_ceil_mode_from_output_sizes(op, graph)
        } else {
            op.attributes
                .get("roundingType")
                .and_then(|v| v.as_str())
                .map(|rounding_type| {
                    if rounding_type.eq_ignore_ascii_case("ceil") {
                        1
                    } else {
                        0
                    }
                })
        } {
            Self::add_int_attribute(&mut attributes, "ceil_mode", ceil_mode);
        }

        attributes
    }

    /// Helper: Add an integer array attribute using shared AttrBuilder
    fn add_ints_attribute(attributes: &mut Vec<AttributeProto>, name: &str, values: Vec<i64>) {
        if !values.is_empty() {
            let builder = AttrBuilder::new().add_ints(name, values);
            attributes.extend(builder.build());
        }
    }

    /// Helper: Add an integer attribute using shared AttrBuilder
    fn add_int_attribute(attributes: &mut Vec<AttributeProto>, name: &str, value: i64) {
        let builder = AttrBuilder::new().add_int(name, value);
        attributes.extend(builder.build());
    }

    /// Create ONNX attributes for conv2d operation
    fn create_conv2d_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON using helpers
        if let Some(strides) = Self::parse_i64_array_any(op, &["strides", "stride"]) {
            Self::add_ints_attribute(&mut attributes, "strides", strides);
        }
        if let Some(dilations) = Self::parse_i64_array_any(op, &["dilations", "dilation"]) {
            Self::add_ints_attribute(&mut attributes, "dilations", dilations);
        }
        if let Some(pads) = Self::parse_pads_attr(op) {
            Self::add_ints_attribute(&mut attributes, "pads", pads);
        }
        if let Some(groups) = op.attributes.get("groups").and_then(|v| v.as_u64()) {
            Self::add_int_attribute(&mut attributes, "group", groups as i64); // Note: ONNX uses "group" not "groups"
        }

        attributes
    }

    /// Create ONNX attributes for convTranspose2d operation
    fn create_conv_transpose2d_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON using helpers
        if let Some(strides) = Self::parse_i64_array_any(op, &["strides", "stride"]) {
            Self::add_ints_attribute(&mut attributes, "strides", strides);
        }
        if let Some(dilations) = Self::parse_i64_array_any(op, &["dilations", "dilation"]) {
            Self::add_ints_attribute(&mut attributes, "dilations", dilations);
        }
        let pads_for_output_shape = Self::parse_pads_attr(op);
        if let Some(pads) = pads_for_output_shape.clone() {
            Self::add_ints_attribute(&mut attributes, "pads", pads);
        }
        if let Some(output_padding) =
            Self::parse_i64_array_any(op, &["outputPadding", "output_padding"])
        {
            Self::add_ints_attribute(&mut attributes, "output_padding", output_padding);
        }
        if let Some(output_shape) = Self::parse_i64_array_any(op, &["outputSizes", "output_shape"])
        {
            // ONNX ConvTranspose with both asymmetric pads and output_shape can choose
            // a different side-alignment than WebNN's same-upper behavior.
            // In that case, rely on pads/strides/kernel formula to determine output size.
            let has_asymmetric_pads = pads_for_output_shape
                .as_ref()
                .map(|p| p.len() == 4 && (p[0] != p[2] || p[1] != p[3]))
                .unwrap_or(false);
            if !has_asymmetric_pads {
                Self::add_ints_attribute(&mut attributes, "output_shape", output_shape);
            }
        }
        if let Some(groups) = op.attributes.get("groups").and_then(|v| v.as_u64()) {
            Self::add_int_attribute(&mut attributes, "group", groups as i64); // Note: ONNX uses "group" not "groups"
        }

        attributes
    }

    /// Create ONNX attributes for pool2d operations
    fn create_pool2d_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON using helpers
        if let Some(kernel_shape) = Self::parse_i64_array_any(
            op,
            &[
                "windowDimensions",
                "window_dimensions",
                "windowdimensions",
                "kernelShape",
                "kernel_shape",
            ],
        ) {
            Self::add_ints_attribute(&mut attributes, "kernel_shape", kernel_shape);
        }
        if let Some(strides) = Self::parse_i64_array_any(op, &["strides", "stride"]) {
            Self::add_ints_attribute(&mut attributes, "strides", strides);
        }
        if let Some(dilations) = Self::parse_i64_array_any(op, &["dilations", "dilation"]) {
            Self::add_ints_attribute(&mut attributes, "dilations", dilations);
        }
        if let Some(pads) = Self::parse_pads_attr(op) {
            Self::add_ints_attribute(&mut attributes, "pads", pads);
        }

        attributes
    }

    /// Create ONNX attributes for reduction operations
    fn create_reduce_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON using helpers
        if let Some(axes) = Self::parse_i64_array(op, "axes") {
            Self::add_ints_attribute(&mut attributes, "axes", axes);
        }
        if let Some(keep_dims) = op
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
        {
            Self::add_int_attribute(&mut attributes, "keepdims", if keep_dims { 1 } else { 0 });
        }

        attributes
    }

    fn create_cast_node(
        node_name: &str,
        input: String,
        output: String,
        to_data_type: ProtoDataType,
    ) -> NodeProto {
        NodeProto {
            input: vec![input],
            output: vec![output],
            name: node_name.to_string(),
            op_type: "Cast".to_string(),
            attribute: vec![AttributeProto {
                name: "to".to_string(),
                r#type: AttributeType::Int as i32,
                i: to_data_type as i64,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// Create ONNX attributes for squeeze/unsqueeze operations
    fn create_squeeze_unsqueeze_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(axes) = Self::parse_i64_array(op, "axes") {
            Self::add_ints_attribute(&mut attributes, "axes", axes);
        }

        attributes
    }

    /// Create ONNX attributes for argMax/argMin operations
    fn create_arg_reduce_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(axis) = op
            .attributes
            .get("axis")
            .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
        {
            attributes.push(AttributeProto {
                name: "axis".to_string(),
                r#type: AttributeType::Int as i32,
                i: axis,
                ..Default::default()
            });
        }

        let keep_dims = op
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        attributes.push(AttributeProto {
            name: "keepdims".to_string(), // ONNX uses "keepdims" not "keepDimensions"
            r#type: AttributeType::Int as i32,
            i: if keep_dims { 1 } else { 0 },
            ..Default::default()
        });

        // Note: outputDataType is handled by the output tensor's data type, not as an attribute

        attributes
    }

    /// Create ONNX attributes for concat operation
    fn create_concat_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Concat requires an axis attribute in ONNX
        if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_i64()) {
            attributes.push(AttributeProto {
                name: "axis".to_string(),
                r#type: AttributeType::Int as i32,
                i: axis,
                ..Default::default()
            });
        }

        attributes
    }

    fn create_softmax_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();
        let axis = op
            .attributes
            .get("axis")
            .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
            .unwrap_or(-1);
        attributes.push(AttributeProto {
            name: "axis".to_string(),
            r#type: AttributeType::Int as i32,
            i: axis,
            ..Default::default()
        });
        attributes
    }

    /// Create ONNX attributes for gather operation
    fn create_gather_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();
        if let Some(axis) = op
            .attributes
            .get("axis")
            .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
        {
            attributes.push(AttributeProto {
                name: "axis".to_string(),
                r#type: AttributeType::Int as i32,
                i: axis,
                ..Default::default()
            });
        }
        attributes
    }

    /// Create ONNX attributes for transpose operation
    fn create_transpose_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(perm) = Self::parse_i64_array(op, "permutation") {
            Self::add_ints_attribute(&mut attributes, "perm", perm);
        }

        attributes
    }

    /// Create ONNX attributes for cast operation
    fn create_cast_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(to_type) = op.attributes.get("to").and_then(|v| v.as_str()) {
            // Convert string data type to ONNX data type code
            let type_code = match to_type.to_ascii_lowercase().as_str() {
                "float32" => ProtoDataType::Float as i64,
                "float16" => ProtoDataType::Float16 as i64,
                "int32" => ProtoDataType::Int32 as i64,
                "uint32" => ProtoDataType::Uint32 as i64,
                "int8" => ProtoDataType::Int8 as i64,
                "uint8" => ProtoDataType::Uint8 as i64,
                "int64" => ProtoDataType::Int64 as i64,
                "int4" | "uint4" => ProtoDataType::Undefined as i64,
                _ => ProtoDataType::Undefined as i64,
            };

            attributes.push(AttributeProto {
                name: "to".to_string(),
                r#type: AttributeType::Int as i32,
                i: type_code,
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for scatterElements operation
    fn create_scatter_elements_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_i64()) {
            attributes.push(AttributeProto {
                name: "axis".to_string(),
                r#type: AttributeType::Int as i32,
                i: axis,
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for tile operation
    fn create_tile_attributes(_op: &Operation) -> Vec<AttributeProto> {
        // For Tile operation, repetitions is provided as a separate input tensor in ONNX
        // Not as an attribute, so we return empty attributes
        Vec::new()
    }

    /// Create ONNX attributes for triangular operation
    fn create_triangular_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(upper) = op.attributes.get("upper").and_then(|v| v.as_bool()) {
            attributes.push(AttributeProto {
                name: "upper".to_string(),
                r#type: AttributeType::Int as i32,
                i: if upper { 1 } else { 0 },
                ..Default::default()
            });
        }

        if let Some(diagonal) = op.attributes.get("diagonal").and_then(|v| v.as_i64()) {
            attributes.push(AttributeProto {
                name: "k".to_string(), // ONNX uses "k" for diagonal offset
                r#type: AttributeType::Int as i32,
                i: diagonal,
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for hardSigmoid operation
    fn create_hardsigmoid_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "alpha".to_string(),
                r#type: AttributeType::Float as i32,
                f: alpha as f32,
                ..Default::default()
            });
        }

        if let Some(beta) = op.attributes.get("beta").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "beta".to_string(),
                r#type: AttributeType::Float as i32,
                f: beta as f32,
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for hardSwish operation
    fn create_hardswish_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "alpha".to_string(),
                r#type: AttributeType::Float as i32,
                f: alpha as f32,
                ..Default::default()
            });
        }

        if let Some(beta) = op.attributes.get("beta").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "beta".to_string(),
                r#type: AttributeType::Float as i32,
                f: beta as f32,
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for elu operation
    fn create_elu_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "alpha".to_string(),
                r#type: AttributeType::Float as i32,
                f: alpha as f32,
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for leakyRelu operation
    fn create_leakyrelu_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "alpha".to_string(),
                r#type: AttributeType::Float as i32,
                f: alpha as f32,
                ..Default::default()
            });
        }

        attributes
    }

    /// Clamp operation doesn't use attributes - min/max are inputs in opset 11+
    /// Handled in convert() method as special case
    ///
    /// Create ONNX attributes for gemm operation
    fn create_gemm_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "alpha".to_string(),
                r#type: AttributeType::Float as i32,
                f: alpha as f32,
                ..Default::default()
            });
        }

        if let Some(beta) = op.attributes.get("beta").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "beta".to_string(),
                r#type: AttributeType::Float as i32,
                f: beta as f32,
                ..Default::default()
            });
        }

        if let Some(a_transpose) = op.attributes.get("aTranspose").and_then(|v| v.as_bool()) {
            attributes.push(AttributeProto {
                name: "transA".to_string(),
                r#type: AttributeType::Int as i32,
                i: if a_transpose { 1 } else { 0 },
                ..Default::default()
            });
        }

        if let Some(b_transpose) = op.attributes.get("bTranspose").and_then(|v| v.as_bool()) {
            attributes.push(AttributeProto {
                name: "transB".to_string(),
                r#type: AttributeType::Int as i32,
                i: if b_transpose { 1 } else { 0 },
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for layerNormalization operation
    fn create_layernorm_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Add epsilon attribute
        let epsilon = op
            .attributes
            .get("epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);
        attributes.push(AttributeProto {
            name: "epsilon".to_string(),
            r#type: AttributeType::Float as i32,
            f: epsilon as f32,
            ..Default::default()
        });

        // Add axis attribute
        // ONNX LayerNormalization normalizes from axis to the end (consecutive dimensions)
        // WebNN allows arbitrary axes, so we need to check compatibility
        let axes = op
            .attributes
            .get("axes")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
            .unwrap_or_else(|| vec![-1]);

        // For now, use the first axis (or -1 for last dimension)
        // TODO: Validate that axes are consecutive and end at last dimension
        // If not, we should emulate using primitive operations (like Chromium does)
        let axis = axes.first().copied().unwrap_or(-1);
        attributes.push(AttributeProto {
            name: "axis".to_string(),
            r#type: AttributeType::Int as i32,
            i: axis,
            ..Default::default()
        });

        attributes
    }

    /// Create ONNX attributes for batchNormalization or instanceNormalization
    fn create_normalization_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Add epsilon attribute
        let epsilon = op
            .attributes
            .get("epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);
        attributes.push(AttributeProto {
            name: "epsilon".to_string(),
            r#type: AttributeType::Float as i32,
            f: epsilon as f32,
            ..Default::default()
        });

        attributes
    }

    fn create_operation_attributes(op: &Operation) -> Vec<AttributeProto> {
        let op_type = op.op_type.to_ascii_lowercase();
        if op_type == "conv2d" {
            Self::create_conv2d_attributes(op)
        } else if op_type == "convtranspose2d" {
            Self::create_conv_transpose2d_attributes(op)
        } else if op_type == "averagepool2d" || op_type == "maxpool2d" {
            Self::create_pool2d_attributes(op)
        } else if op_type.starts_with("reduce") {
            Self::create_reduce_attributes(op)
        } else if op_type == "squeeze" || op_type == "unsqueeze" {
            Self::create_squeeze_unsqueeze_attributes(op)
        } else if op_type == "argmax" || op_type == "argmin" {
            Self::create_arg_reduce_attributes(op)
        } else if op_type == "concat" {
            Self::create_concat_attributes(op)
        } else if op_type == "gather" {
            Self::create_gather_attributes(op)
        } else if op_type == "transpose" {
            Self::create_transpose_attributes(op)
        } else if op_type == "softmax" {
            Self::create_softmax_attributes(op)
        } else if op_type == "cast" {
            Self::create_cast_attributes(op)
        } else if op_type == "scatterelements" {
            Self::create_scatter_elements_attributes(op)
        } else if op_type == "tile" {
            Self::create_tile_attributes(op)
        } else if op_type == "triangular" {
            Self::create_triangular_attributes(op)
        } else if op_type == "hardsigmoid" {
            Self::create_hardsigmoid_attributes(op)
        } else if op_type == "hardswish" {
            Self::create_hardswish_attributes(op)
        } else if op_type == "elu" {
            Self::create_elu_attributes(op)
        } else if op_type == "leakyrelu" {
            Self::create_leakyrelu_attributes(op)
        } else if op_type == "gemm" {
            Self::create_gemm_attributes(op)
        } else if op_type == "layernormalization" {
            Self::create_layernorm_attributes(op)
        } else if op_type == "batchnormalization" || op_type == "instancenormalization" {
            Self::create_normalization_attributes(op)
        } else if op_type == "clamp" {
            // Clamp (Clip in ONNX opset 11+) uses inputs for min/max, not attributes
            Vec::new()
        } else {
            Vec::new()
        }
    }
}

impl crate::converters::GraphConverter for OnnxConverter {
    fn format(&self) -> &'static str {
        "onnx"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        if !crate::graph::dynamic_inputs_enabled() && graph.has_dynamic_dimensions() {
            return Err(GraphError::DynamicInputsFeatureDisabled);
        }

        debug_print!("[DEBUG] Starting ONNX conversion");
        debug_print!("  Total operations: {}", graph.operations.len());
        let expand_count = graph
            .operations
            .iter()
            .filter(|op| op.op_type == "expand")
            .count();
        debug_print!("  Expand operations: {}", expand_count);

        let mut initializers = Vec::new();
        let mut inputs_val = Vec::new();
        let mut outputs_val = Vec::new();
        let value_infos = Vec::new();
        let mut skipped_inputs = std::collections::HashSet::new(); // Track skipped empty KV inputs
        let operand_remapping: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new(); // Map skipped outputs to replacements

        for &id in &graph.input_operands {
            let operand = graph.operand(id).ok_or_else(|| {
                debug_print!(
                    "[DEBUG] Missing input operand {} while building ONNX graph",
                    id
                );
                Self::invalid_operand("graph input lookup", id, None)
            })?;

            // Skip KV cache inputs with empty dimensions (past_sequence_length=0)
            // These inputs are never used in the computation - they're just concatenated
            // with new KV, and concat(empty, new) = new.
            let input_name = operand_name(graph, id);
            let has_empty_dimension = operand.descriptor.static_or_max_shape().contains(&0);
            let is_kv_input = input_name.starts_with("past_key_values_");

            // Debug: print all KV input shapes
            if is_kv_input {
                debug_print!(
                    "[ONNX CONVERTER] KV input: {} shape={:?} has_empty={}",
                    input_name,
                    operand.descriptor.shape,
                    has_empty_dimension
                );
            }

            if has_empty_dimension && is_kv_input {
                debug_print!(
                    "[DEBUG] Skipping empty KV cache input: {} (shape: {:?})",
                    input_name,
                    operand.descriptor.shape
                );
                skipped_inputs.insert(id);
                continue;
            }

            inputs_val.push(value_info(&input_name, &operand.descriptor));
        }

        // Sort outputs: "logits" first, then alphabetically by name
        // This ensures ONNX models have logits at output 0 (expected by users)
        let mut sorted_outputs: Vec<u32> = graph.output_operands.clone();
        sorted_outputs.sort_by_key(|&id| {
            let operand = graph.operand(id);
            let name = operand.and_then(|op| op.name.as_deref()).unwrap_or("");
            // Sort key: (priority, name)
            // "logits" gets priority 0 (first), everything else gets priority 1 (alphabetically)
            if name == "logits" {
                (0, String::new())
            } else {
                (1, name.to_string())
            }
        });

        for &id in &sorted_outputs {
            let operand = graph.operand(id).ok_or_else(|| {
                debug_print!(
                    "[DEBUG] Missing output operand {} while building ONNX graph",
                    id
                );
                Self::invalid_operand("graph output lookup", id, None)
            })?;

            // Logic operations output uint8 in WebNN (matching Chromium)
            // ONNX models will correctly use uint8 for logical operation outputs
            // The executor handles uint8 → f32 conversion for Python compatibility
            outputs_val.push(value_info(&operand_name(graph, id), &operand.descriptor));
        }

        // Build type overrides for ops where output type must match an input (e.g., expand)
        let mut type_overrides: std::collections::HashMap<u32, DataType> =
            std::collections::HashMap::new();
        let mut shape_overrides: std::collections::HashMap<u32, Vec<u32>> =
            std::collections::HashMap::new();
        let mut unsqueeze_like_outputs: std::collections::HashSet<u32> =
            std::collections::HashSet::new();
        let mut operand_shapes: std::collections::HashMap<u32, Vec<u32>> =
            std::collections::HashMap::new();
        let mut name_to_id: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();

        // Seed operand_shapes with known operand descriptors
        for (idx, operand) in graph.operands.iter().enumerate() {
            if !operand.descriptor.shape.is_empty() {
                operand_shapes.insert(idx as u32, operand.descriptor.static_or_max_shape());
            }
            if let Some(name) = &operand.name {
                name_to_id.insert(name.clone(), idx as u32);
            }
        }

        // Helper to read constant int64 values by operand name
        let get_const_i64 = |name: &str| -> Option<Vec<i64>> {
            let id = name_to_id.get(name)?;
            let handle = graph.constant_operand_ids_to_handles.get(id)?;
            let bytes = &handle.data;
            if bytes.len() % 8 != 0 {
                return None;
            }
            Some(
                bytes
                    .chunks(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            )
        };

        for op in &graph.operations {
            // Preserve input type for shape-only transforms regardless of shape inference success.
            if (op.op_type.eq_ignore_ascii_case("unsqueeze")
                || op.op_type.eq_ignore_ascii_case("squeeze"))
                && let (Some(output_id), Some(&input_id)) =
                    (op.output_operand, op.input_operands.first())
            {
                unsqueeze_like_outputs.insert(output_id);
                let input_type = type_overrides.get(&input_id).copied().or_else(|| {
                    graph
                        .operand(input_id)
                        .map(|operand| operand.descriptor.data_type)
                });
                if let Some(dtype) = input_type {
                    type_overrides.insert(output_id, dtype);
                }
            }

            if op.op_type.eq_ignore_ascii_case("expand") {
                if let (Some(&input_id), Some(output_id)) =
                    (op.input_operands.first(), op.output_operand)
                    && let Some(input_operand) = graph.operand(input_id)
                {
                    type_overrides.insert(output_id, input_operand.descriptor.data_type);

                    // Check for newShape attribute first
                    if let Some(new_shape_attr) = op.attributes.get("newShape") {
                        if let Some(shape_dims) = Self::parse_dimension_array(new_shape_attr) {
                            let shape: Vec<u32> =
                                shape_dims.iter().map(get_static_or_max_size).collect();
                            if !shape.is_empty() {
                                shape_overrides.insert(output_id, shape.clone());
                                operand_shapes.insert(output_id, shape);
                            }
                        }
                    }
                    // Fall back to shape from second input operand (if present)
                    else if op.input_operands.len() >= 2
                        && let Some(shape) = operand_shapes.get(&op.input_operands[1])
                    {
                        shape_overrides.insert(output_id, shape.clone());
                        operand_shapes.insert(output_id, shape.clone());
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("shape") {
                if let Some(output_id) = op.output_operand {
                    type_overrides.insert(output_id, DataType::Int64);
                }
            } else if op.op_type.eq_ignore_ascii_case("where") {
                if let (Some(output_id), Some(val_input_id)) =
                    (op.output_operand, op.input_operands.get(1))
                    && let Some(input_operand) = graph.operand(*val_input_id)
                {
                    type_overrides.insert(output_id, input_operand.descriptor.data_type);
                }
            } else if op.op_type.eq_ignore_ascii_case("slice") {
                if let (Some(&input_id), Some(output_id)) =
                    (op.input_operands.first(), op.output_operand)
                    && let Some(mut in_shape) = operand_shapes.get(&input_id).cloned()
                {
                    // Preserve input dtype for the slice output
                    if let Some(input_operand) = graph.operand(input_id) {
                        type_overrides.insert(output_id, input_operand.descriptor.data_type);
                    }
                    let axes = op
                        .attributes
                        .get("axes")
                        .and_then(|v| v.as_str())
                        .and_then(get_const_i64);
                    let starts = op
                        .attributes
                        .get("starts")
                        .and_then(|v| v.as_str())
                        .and_then(get_const_i64);
                    let ends = op
                        .attributes
                        .get("ends")
                        .and_then(|v| v.as_str())
                        .and_then(get_const_i64);
                    let steps = op
                        .attributes
                        .get("steps")
                        .and_then(|v| v.as_str())
                        .and_then(get_const_i64);

                    if let (Some(axes), Some(starts), Some(ends)) = (axes, starts, ends) {
                        let steps_vec = steps.unwrap_or_else(|| vec![1; axes.len()]);
                        let len = axes
                            .len()
                            .min(starts.len())
                            .min(ends.len())
                            .min(steps_vec.len());
                        for i in 0..len {
                            let axis = axes[i] as isize;
                            if axis < 0 || (axis as usize) >= in_shape.len() {
                                continue;
                            }
                            let step = steps_vec[i];
                            if step == 0 {
                                continue;
                            }
                            let start = starts[i];
                            let end = ends[i];
                            let dim = in_shape[axis as usize] as i64;
                            let s = if start < 0 { dim + start } else { start }.max(0);
                            let e = if end < 0 { dim + end } else { end }.min(dim);
                            let span = (e - s + (step.abs() - 1)) / step.abs();
                            if span > 0 {
                                in_shape[axis as usize] = span as u32;
                            }
                        }
                        shape_overrides.insert(output_id, in_shape.clone());
                        operand_shapes.insert(output_id, in_shape);
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("cos")
                || op.op_type.eq_ignore_ascii_case("sin")
                || op.op_type.eq_ignore_ascii_case("tan")
                || op.op_type.eq_ignore_ascii_case("exp")
                || op.op_type.eq_ignore_ascii_case("log")
                || op.op_type.eq_ignore_ascii_case("abs")
                || op.op_type.eq_ignore_ascii_case("neg")
                || op.op_type.eq_ignore_ascii_case("sqrt")
                || op.op_type.eq_ignore_ascii_case("relu")
                || op.op_type.eq_ignore_ascii_case("sigmoid")
                || op.op_type.eq_ignore_ascii_case("tanh")
                || op.op_type.eq_ignore_ascii_case("cast")
            {
                // Track unary element-wise operations (preserve input shape and type)
                if let Some(output_id) = op.output_operand
                    && let Some(&input_id) = op.input_operands.first()
                {
                    let output_name = graph
                        .operand(output_id)
                        .and_then(|op| op.name.as_ref())
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");

                    // Preserve shape from input
                    if let Some(input_shape) = operand_shapes.get(&input_id) {
                        debug_print!(
                            "[UNARY DEBUG] {} op {} preserves shape {:?} from input {}",
                            op.op_type,
                            output_name,
                            input_shape,
                            input_id
                        );
                        shape_overrides.insert(output_id, input_shape.clone());
                        operand_shapes.insert(output_id, input_shape.clone());
                    } else {
                        debug_print!(
                            "[UNARY WARNING] {} op {} has no input shape for input {}",
                            op.op_type,
                            output_name,
                            input_id
                        );
                    }

                    // Preserve type from input (except Cast which changes type)
                    if !op.op_type.eq_ignore_ascii_case("cast") {
                        let input_type = type_overrides
                            .get(&input_id)
                            .copied()
                            .or_else(|| graph.operand(input_id).map(|op| op.descriptor.data_type));

                        if let Some(dtype) = input_type {
                            type_overrides.insert(output_id, dtype);
                        }
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("add")
                || op.op_type.eq_ignore_ascii_case("sub")
                || op.op_type.eq_ignore_ascii_case("mul")
                || op.op_type.eq_ignore_ascii_case("div")
                || op.op_type.eq_ignore_ascii_case("pow")
            {
                // Track binary element-wise operation output shapes (use broadcasting)
                if let Some(output_id) = op.output_operand
                    && op.input_operands.len() >= 2
                {
                    // Try to compute broadcast shape from inputs
                    if let (Some(lhs), Some(rhs)) = (
                        operand_shapes.get(&op.input_operands[0]),
                        operand_shapes.get(&op.input_operands[1]),
                    ) && let Ok(result_shape) = broadcast_shapes(lhs, rhs)
                    {
                        shape_overrides.insert(output_id, result_shape.clone());
                        operand_shapes.insert(output_id, result_shape);
                    }

                    // Preserve type from first input (binary ops typically preserve input type)
                    if let Some(&first_input_id) = op.input_operands.first() {
                        let input_type =
                            type_overrides.get(&first_input_id).copied().or_else(|| {
                                graph
                                    .operand(first_input_id)
                                    .map(|op| op.descriptor.data_type)
                            });

                        if let Some(dtype) = input_type {
                            type_overrides.insert(output_id, dtype);
                        }
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("matmul") {
                // Track matmul output shapes
                if let Some(output_id) = op.output_operand
                    && op.input_operands.len() == 2
                {
                    let output_name = graph
                        .operand(output_id)
                        .and_then(|op| op.name.as_ref())
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");

                    // Get input shapes
                    let lhs_shape = operand_shapes.get(&op.input_operands[0]);
                    let rhs_shape = operand_shapes.get(&op.input_operands[1]);

                    if let (Some(lhs), Some(rhs)) = (lhs_shape, rhs_shape) {
                        if let Ok(out_shape) = infer_matmul_shape(lhs, rhs) {
                            debug_print!(
                                "[MATMUL DEBUG] Matmul {} tracked output shape {:?} from inputs {:?} @ {:?}",
                                output_name,
                                out_shape,
                                lhs,
                                rhs
                            );
                            shape_overrides.insert(output_id, out_shape.clone());
                            operand_shapes.insert(output_id, out_shape);
                        } else {
                            debug_print!(
                                "[MATMUL WARNING] Matmul {} failed to infer shape from inputs {:?} @ {:?}",
                                output_name,
                                lhs,
                                rhs
                            );
                        }
                    } else {
                        debug_print!(
                            "[MATMUL WARNING] Matmul {} missing input shapes: lhs={} rhs={}",
                            output_name,
                            lhs_shape.is_some(),
                            rhs_shape.is_some()
                        );
                    }

                    // Preserve type from first input
                    let input_type =
                        type_overrides
                            .get(&op.input_operands[0])
                            .copied()
                            .or_else(|| {
                                graph
                                    .operand(op.input_operands[0])
                                    .map(|op| op.descriptor.data_type)
                            });

                    if let Some(dtype) = input_type {
                        type_overrides.insert(output_id, dtype);
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("concat") {
                // Track concat output shapes
                if let Some(output_id) = op.output_operand
                    && op.input_operands.len() >= 2
                {
                    let output_name = graph
                        .operand(output_id)
                        .and_then(|op| op.name.as_ref())
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");

                    // Get axis attribute
                    let axis = op
                        .attributes
                        .get("axis")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);

                    // Get input shapes
                    let input_shapes: Vec<_> = op
                        .input_operands
                        .iter()
                        .filter_map(|id| operand_shapes.get(id))
                        .collect();

                    if input_shapes.len() == op.input_operands.len() && !input_shapes.is_empty() {
                        let rank = input_shapes[0].len() as i64;
                        let mut axis = axis;
                        if axis < 0 {
                            axis += rank;
                        }
                        if axis >= 0 && (axis as usize) < input_shapes[0].len() {
                            let mut out_shape = input_shapes[0].clone();
                            let concat_axis = axis as usize;
                            for shape in &input_shapes[1..] {
                                out_shape[concat_axis] += shape[concat_axis];
                            }
                            debug_print!(
                                "[CONCAT DEBUG] Concat {} tracked output shape {:?}",
                                output_name,
                                out_shape
                            );
                            shape_overrides.insert(output_id, out_shape.clone());
                            operand_shapes.insert(output_id, out_shape);
                        }
                    } else {
                        debug_print!(
                            "[CONCAT WARNING] Concat {} missing input shapes: have {}/{} inputs",
                            output_name,
                            input_shapes.len(),
                            op.input_operands.len()
                        );
                    }

                    // Preserve type from first input (concat requires all inputs have same type)
                    if let Some(&first_input_id) = op.input_operands.first() {
                        // Check if we have a type override for the first input
                        let input_type =
                            type_overrides.get(&first_input_id).copied().or_else(|| {
                                // Fall back to descriptor type
                                graph
                                    .operand(first_input_id)
                                    .map(|op| op.descriptor.data_type)
                            });

                        if let Some(dtype) = input_type {
                            type_overrides.insert(output_id, dtype);
                        }
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("unsqueeze") {
                // Track unsqueeze output shapes (adds dimensions)
                if let Some(output_id) = op.output_operand
                    && let Some(&input_id) = op.input_operands.first()
                    && let Some(input_shape) = operand_shapes.get(&input_id)
                    && let Some(axes_val) = op.attributes.get("axes")
                {
                    let axes_i64: Vec<i64> = if let Some(arr) = axes_val.as_array() {
                        arr.iter().filter_map(|v| v.as_i64()).collect()
                    } else {
                        vec![]
                    };

                    if !axes_i64.is_empty() {
                        use crate::shape_inference::infer_unsqueeze_shape;
                        let axes_u32: Vec<u32> = axes_i64
                            .iter()
                            .map(|&a| {
                                if a < 0 {
                                    ((input_shape.len() as i64 + 1) + a) as u32
                                } else {
                                    a as u32
                                }
                            })
                            .collect();

                        if let Ok(out_shape) = infer_unsqueeze_shape(input_shape, &axes_u32) {
                            shape_overrides.insert(output_id, out_shape.clone());
                            operand_shapes.insert(output_id, out_shape);

                            // Preserve input type for unsqueeze output
                            if let Some(input_operand) = graph.operand(input_id) {
                                type_overrides
                                    .insert(output_id, input_operand.descriptor.data_type);
                            }
                        }
                    }
                }
            }
            // Reshape: if newShape is static, set output shape
            else if op.op_type.eq_ignore_ascii_case("reshape") {
                if let Some(output_id) = op.output_operand
                    && let Some(new_shape_val) = op.attributes.get("newShape")
                    && let Some(shape_dims) = Self::parse_dimension_array(new_shape_val)
                {
                    let shape: Vec<u32> = shape_dims.iter().map(get_static_or_max_size).collect();
                    if !shape.is_empty() {
                        shape_overrides.insert(output_id, shape.clone());
                        operand_shapes.insert(output_id, shape);
                    }
                }
            }
            // Transpose: derive output shape from permutation (default reverse)
            else if op.op_type.eq_ignore_ascii_case("transpose") {
                if let (Some(&input_id), Some(output_id)) =
                    (op.input_operands.first(), op.output_operand)
                    && let Some(input_shape) = operand_shapes.get(&input_id).cloned()
                {
                    let perm: Option<Vec<u32>> = op
                        .attributes
                        .get("permutation")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| {
                                    v.as_i64()
                                        .or_else(|| v.as_u64().map(|u| u as i64))
                                        .map(|n| n as u32)
                                })
                                .collect()
                        });

                    if let Ok(out_shape) = infer_transpose_shape(&input_shape, perm.as_deref()) {
                        shape_overrides.insert(output_id, out_shape.clone());
                        operand_shapes.insert(output_id, out_shape);
                    }
                }
            }
            // Gather: infer output shape from data/indices if available
            else if op.op_type.eq_ignore_ascii_case("gather")
                && let Some(output_id) = op.output_operand
                && op.input_operands.len() >= 2
            {
                let data_shape = operand_shapes.get(&op.input_operands[0]);
                let indices_shape = operand_shapes.get(&op.input_operands[1]);
                let axis = op
                    .attributes
                    .get("axis")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                if let (Some(data_shape), Some(indices_shape)) = (data_shape, indices_shape) {
                    let rank = data_shape.len() as i64;
                    let mut axis = axis;
                    if axis < 0 {
                        axis += rank;
                    }
                    if axis >= 0 && (axis as usize) < data_shape.len() {
                        let mut out_shape = indices_shape.clone();
                        out_shape.extend_from_slice(&data_shape[(axis as usize + 1)..]);
                        shape_overrides.insert(output_id, out_shape.clone());
                        operand_shapes.insert(output_id, out_shape);
                    }
                }
            }

            // Update operand_shapes for outputs where we inferred a shape override
            if let Some(out_id) = op.output_operand
                && let Some(shape) = shape_overrides.get(&out_id)
            {
                operand_shapes.insert(out_id, shape.clone());
            }
        }

        for (id, data) in &graph.constant_operand_ids_to_handles {
            let operand = graph.operand(*id).ok_or_else(|| {
                debug_print!(
                    "[DEBUG] Missing constant operand {} while building initializers",
                    id
                );
                Self::invalid_operand("initializer lookup", *id, None)
            })?;

            // Handle zero-length constants by creating zero-filled tensors
            // This is a defensive measure for malformed models where constants have no data
            let tensor_proto = if data.data.is_empty() {
                let element_count: usize = operand
                    .descriptor
                    .static_or_max_shape()
                    .iter()
                    .map(|&d| d as usize)
                    .product();
                let dtype = Self::data_type_code(operand.descriptor.data_type);

                // For Int64, use int64_data field; for other types, use raw_data with zeros
                match operand.descriptor.data_type {
                    DataType::Int64 => TensorProto {
                        name: operand_name(graph, *id),
                        data_type: dtype as i32,
                        dims: operand
                            .descriptor
                            .static_or_max_shape()
                            .iter()
                            .map(|d| *d as i64)
                            .collect(),
                        int64_data: vec![0i64; element_count],
                        ..Default::default()
                    },
                    DataType::Int32 => TensorProto {
                        name: operand_name(graph, *id),
                        data_type: dtype as i32,
                        dims: operand
                            .descriptor
                            .static_or_max_shape()
                            .iter()
                            .map(|d| *d as i64)
                            .collect(),
                        int32_data: vec![0i32; element_count],
                        ..Default::default()
                    },
                    DataType::Float32 => TensorProto {
                        name: operand_name(graph, *id),
                        data_type: dtype as i32,
                        dims: operand
                            .descriptor
                            .static_or_max_shape()
                            .iter()
                            .map(|d| *d as i64)
                            .collect(),
                        float_data: vec![0f32; element_count],
                        ..Default::default()
                    },
                    _ => {
                        // For other types, create zero-filled raw_data
                        let bytes_per_element = match operand.descriptor.data_type {
                            DataType::Int4 | DataType::Uint4 => 1,
                            DataType::Float16 => 2,
                            DataType::Int8 | DataType::Uint8 => 1,
                            DataType::Uint32 => 4,
                            DataType::Uint64 => 8,
                            _ => 4, // Default to 4 bytes
                        };
                        let zero_data = vec![0u8; element_count * bytes_per_element];
                        TensorProto {
                            name: operand_name(graph, *id),
                            data_type: dtype as i32,
                            dims: operand
                                .descriptor
                                .static_or_max_shape()
                                .iter()
                                .map(|d| *d as i64)
                                .collect(),
                            raw_data: zero_data,
                            ..Default::default()
                        }
                    }
                }
            } else {
                // Normal case: use provided data
                TensorProto {
                    name: operand_name(graph, *id),
                    data_type: Self::data_type_code(operand.descriptor.data_type) as i32,
                    dims: operand
                        .descriptor
                        .static_or_max_shape()
                        .iter()
                        .map(|d| *d as i64)
                        .collect(),
                    raw_data: data.data.clone(),
                    ..Default::default()
                }
            };

            initializers.push(tensor_proto);
        }

        // Generate nodes, inserting Cast nodes for logic operations
        let mut nodes = Vec::new();
        let mut cast_counter = 0;

        if let Some(opd) = graph.operands.get(34) {
            debug_print!(
                "[DEBUG] Operand 34 name={:?} dtype={:?} shape={:?}",
                opd.name,
                opd.descriptor.data_type,
                opd.descriptor.shape
            );
        } else {
            debug_print!("[DEBUG] Operand 34 not present in operands table");
        }

        for (idx, op) in graph.operations.iter().enumerate() {
            // Debug guard: ensure all input operands exist
            for &input_id in &op.input_operands {
                // Resolve remapping first
                let resolved_id = operand_remapping
                    .get(&input_id)
                    .copied()
                    .unwrap_or(input_id);
                if graph.operand(resolved_id).is_none() {
                    let input_name = graph
                        .operands
                        .get(input_id as usize)
                        .and_then(|opd| opd.name.clone())
                        .unwrap_or_else(|| format!("<unnamed:{}>", input_id));
                    debug_print!(
                        "[DEBUG] Missing operand id {} name '{}' for op {} ({}) at index {}. Inputs: {:?}",
                        input_id,
                        input_name,
                        op.label.clone().unwrap_or_else(|| op.op_type.clone()),
                        op.op_type,
                        idx,
                        op.input_operands
                    );
                    debug_print!(
                        "[DEBUG] operands.len()={} valid ids 0..{}",
                        graph.operands.len(),
                        graph.operands.len().saturating_sub(1)
                    );
                    debug_print!(
                        "[DEBUG] Failing op detail: idx={} type={} label={:?} inputs={:?}",
                        idx,
                        op.op_type,
                        op.label,
                        op.input_operands
                    );
                    return Err(Self::invalid_operand(
                        "op input lookup",
                        input_id,
                        Some((op, idx)),
                    ));
                }
            }

            // Replace concat operations with empty KV inputs with Identity nodes
            // For past_sequence_length=0, concat(empty, new) = new, so we just copy the input
            if op.op_type.eq_ignore_ascii_case("concat") {
                let has_skipped_input = op
                    .input_operands
                    .iter()
                    .any(|&id| skipped_inputs.contains(&id));

                // Debug: print all concat ops
                debug_print!(
                    "[ONNX CONVERTER] Concat op idx={} has {} inputs, has_skipped={}",
                    idx,
                    op.input_operands.len(),
                    has_skipped_input
                );

                if has_skipped_input {
                    // Find the non-skipped input (the actual new KV)
                    let remaining_inputs: Vec<_> = op
                        .input_operands
                        .iter()
                        .filter(|&&id| !skipped_inputs.contains(&id))
                        .copied()
                        .collect();

                    debug_print!(
                        "[ONNX CONVERTER]   Remaining inputs: {}",
                        remaining_inputs.len()
                    );

                    if remaining_inputs.len() == 1 {
                        // Perfect case: one skipped (empty past), one remaining (new KV)
                        let output_id = op.output_operand.ok_or_else(|| {
                            Self::invalid_operand("concat output", idx as u32, Some((op, idx)))
                        })?;

                        let input_id = remaining_inputs[0];
                        let resolved_input_id = operand_remapping
                            .get(&input_id)
                            .copied()
                            .unwrap_or(input_id);
                        let input_name = operand_name(graph, resolved_input_id);
                        let output_name = operand_name(graph, output_id);

                        debug_print!(
                            "[ONNX CONVERTER]   Creating Identity node: {} -> {}",
                            input_name,
                            output_name
                        );

                        // Create an Identity node: output = Identity(input)
                        let identity_node = NodeProto {
                            input: vec![input_name],
                            output: vec![output_name.clone()],
                            name: format!("identity_{}", output_id),
                            op_type: "Identity".to_string(),
                            ..Default::default()
                        };

                        nodes.push(identity_node);

                        // Track shape for the output
                        operand_shapes.insert(
                            output_id,
                            graph
                                .operand(resolved_input_id)
                                .map(|opd| opd.descriptor.static_or_max_shape())
                                .unwrap_or_default(),
                        );

                        continue;
                    }
                }
            }

            // WebNN constant() op: encode as initializer, not a node
            if op.op_type.eq_ignore_ascii_case("constant") {
                let output_id = op.output_operand.ok_or_else(|| {
                    Self::invalid_operand("constant output", idx as u32, Some((op, idx)))
                })?;

                // Get constant data: try 'init' attribute first (reference to named constant),
                // then fall back to 'data' attribute (inline base64)
                let data = if let Some(init_ref) =
                    op.attributes.get("init").and_then(|v| v.as_str())
                {
                    // 'init' attribute references a named constant declaration (e.g., "$_name")
                    // The operand name in the graph keeps the '$' prefix
                    debug_print!("[DEBUG] Constant operation with 'init' reference:");
                    debug_print!("  Operation index: {}", idx);
                    debug_print!("  Output operand: {}", output_id);
                    debug_print!("  Init reference: {}", init_ref);
                    debug_print!("  Looking for constant operand named: {}", init_ref);

                    // Find the constant operand with matching name
                    // Note: Named constants from the constants{} section have OperandKind::Constant
                    let const_operand_id = graph
                        .operands
                        .iter()
                        .enumerate()
                        .find(|(_, op)| {
                            op.name.as_deref() == Some(init_ref) && op.kind == OperandKind::Constant
                        })
                        .map(|(id, _)| id as u32)
                        .ok_or_else(|| {
                            debug_print!("[DEBUG] Failed to find constant operand:");
                            debug_print!("  All constant operands:");
                            for (id, op) in graph.operands.iter().enumerate() {
                                if op.kind == OperandKind::Constant {
                                    debug_print!("    ID {}: name={:?}", id, op.name);
                                }
                            }
                            GraphError::ConversionFailed {
                                format: "onnx".to_string(),
                                reason: format!(
                                    "Constant op init='{}' references unknown constant operand",
                                    init_ref
                                ),
                            }
                        })?;

                    // Look up the constant data
                    graph
                        .constant_operand_ids_to_handles
                        .get(&const_operand_id)
                        .map(|const_data| const_data.data.clone())
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!(
                                "Constant op init='{}' found operand {} but no data in constant_operand_ids_to_handles",
                                init_ref, const_operand_id
                            ),
                        })?
                } else if let Some(data_b64) = op.attributes.get("data").and_then(|v| v.as_str()) {
                    // 'data' attribute contains inline base64-encoded data
                    STANDARD
                        .decode(data_b64)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!("Constant op base64 decode failed: {}", e),
                        })?
                } else {
                    // Neither 'init' nor 'data' found
                    debug_print!("[DEBUG] Constant operation missing 'data' or 'init' attribute:");
                    debug_print!("  Operation index: {}", idx);
                    debug_print!("  Output operand: {}", output_id);
                    if let Some(obj) = op.attributes.as_object() {
                        debug_print!(
                            "  Available attributes: {:?}",
                            obj.keys().collect::<Vec<_>>()
                        );
                    } else {
                        debug_print!("  Attributes: {:?}", op.attributes);
                    }
                    debug_print!("  Label: {:?}", op.label);
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: "Constant op missing both 'data' and 'init' attributes".to_string(),
                    });
                };

                let dtype_str = op
                    .attributes
                    .get("dataType")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: "Constant op missing 'dataType' attribute".to_string(),
                    })?;
                let data_type = match dtype_str.to_ascii_lowercase().as_str() {
                    "float32" => DataType::Float32,
                    "float16" => DataType::Float16,
                    "int32" => DataType::Int32,
                    "uint32" => DataType::Uint32,
                    "int64" => DataType::Int64,
                    "uint64" => DataType::Uint64,
                    "int8" => DataType::Int8,
                    "uint8" => DataType::Uint8,
                    "int4" => DataType::Int4,
                    "uint4" => DataType::Uint4,
                    other => {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!("Unsupported constant dataType '{}'", other),
                        });
                    }
                };

                let shape: Vec<i64> = op
                    .attributes
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
                            .collect()
                    })
                    .unwrap_or_default();

                initializers.push(TensorProto {
                    name: operand_name(graph, output_id),
                    data_type: Self::data_type_code(data_type) as i32,
                    dims: shape,
                    raw_data: data,
                    ..Default::default()
                });

                continue;
            }

            let op_name = op
                .label
                .clone()
                .unwrap_or_else(|| format!("{}_{}", op.op_type, idx));

            // QuantizeLinear is lowered via primitive ops for broader ORT compatibility and
            // to match WebNN blockwise/per-tensor behavior.
            if op.op_type.eq_ignore_ascii_case("quantizeLinear") {
                let input_id = op.input_operands[0];
                let scale_id = op.input_operands[1];
                let zero_point_id = op.input_operands[2];
                let output_id = op
                    .output_operand
                    .ok_or(GraphError::InvalidConversionOperand { operand: 0 })?;

                let input_shape = operand_shapes.get(&input_id).cloned().unwrap_or_else(|| {
                    graph
                        .operand(input_id)
                        .map(|o| o.descriptor.static_or_max_shape())
                        .unwrap_or_default()
                });
                let scale_shape = operand_shapes.get(&scale_id).cloned().unwrap_or_else(|| {
                    graph
                        .operand(scale_id)
                        .map(|o| o.descriptor.static_or_max_shape())
                        .unwrap_or_default()
                });
                let zero_point_shape =
                    operand_shapes
                        .get(&zero_point_id)
                        .cloned()
                        .unwrap_or_else(|| {
                            graph
                                .operand(zero_point_id)
                                .map(|o| o.descriptor.static_or_max_shape())
                                .unwrap_or_default()
                        });
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("quantizeLinear input lookup", input_id, Some((op, idx)))
                })?;
                let scale_operand = graph.operand(scale_id).ok_or_else(|| {
                    Self::invalid_operand("quantizeLinear scale lookup", scale_id, Some((op, idx)))
                })?;

                fn align_param_with_input(
                    op_name: &str,
                    prefix: &str,
                    name: String,
                    param_shape: &[u32],
                    input_shape: &[u32],
                    nodes: &mut Vec<NodeProto>,
                    initializers: &mut Vec<TensorProto>,
                ) -> String {
                    if input_shape.is_empty()
                        || param_shape.is_empty()
                        || param_shape == input_shape
                        || param_shape.len() != input_shape.len()
                    {
                        return name;
                    }

                    let broadcastable = param_shape
                        .iter()
                        .zip(input_shape.iter())
                        .all(|(&p, &i)| p == 1 || p == i);
                    if broadcastable {
                        return name;
                    }

                    let tileable = param_shape
                        .iter()
                        .zip(input_shape.iter())
                        .all(|(&p, &i)| p > 0 && i % p == 0);
                    if !tileable {
                        return name;
                    }

                    let repeats: Vec<i64> = input_shape
                        .iter()
                        .zip(param_shape.iter())
                        .map(|(&i, &p)| (i / p) as i64)
                        .collect();
                    if repeats.iter().all(|&r| r == 1) {
                        return name;
                    }

                    let expanded_shape: Vec<i64> = param_shape
                        .iter()
                        .flat_map(|&d| [d as i64, 1_i64])
                        .collect();
                    let expanded_shape_name = format!("{}_{}_expand_shape", op_name, prefix);
                    initializers.push(TensorProto {
                        name: expanded_shape_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![expanded_shape.len() as i64],
                        int64_data: expanded_shape,
                        ..Default::default()
                    });

                    let expanded_name = format!("{}_{}_expanded", op_name, prefix);
                    nodes.push(NodeProto {
                        input: vec![name, expanded_shape_name],
                        output: vec![expanded_name.clone()],
                        name: format!("{}_reshape_expand_{}", op_name, prefix),
                        op_type: "Reshape".to_string(),
                        ..Default::default()
                    });

                    let tile_repeats: Vec<i64> = repeats.iter().flat_map(|&r| [1_i64, r]).collect();
                    let tile_repeats_name = format!("{}_{}_tile_repeats", op_name, prefix);
                    initializers.push(TensorProto {
                        name: tile_repeats_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![tile_repeats.len() as i64],
                        int64_data: tile_repeats,
                        ..Default::default()
                    });

                    let tiled_name = format!("{}_{}_tiled", op_name, prefix);
                    nodes.push(NodeProto {
                        input: vec![expanded_name, tile_repeats_name],
                        output: vec![tiled_name.clone()],
                        name: format!("{}_tile_{}", op_name, prefix),
                        op_type: "Tile".to_string(),
                        ..Default::default()
                    });

                    let final_shape: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
                    let final_shape_name = format!("{}_{}_final_shape", op_name, prefix);
                    initializers.push(TensorProto {
                        name: final_shape_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![final_shape.len() as i64],
                        int64_data: final_shape,
                        ..Default::default()
                    });

                    let aligned_name = format!("{}_{}_aligned", op_name, prefix);
                    nodes.push(NodeProto {
                        input: vec![tiled_name, final_shape_name],
                        output: vec![aligned_name.clone()],
                        name: format!("{}_reshape_final_{}", op_name, prefix),
                        op_type: "Reshape".to_string(),
                        ..Default::default()
                    });
                    aligned_name
                }

                let input_name = operand_name(graph, input_id);
                let mut scale_name = operand_name(graph, scale_id);
                let mut zero_point_name = operand_name(graph, zero_point_id);
                scale_name = align_param_with_input(
                    &op_name,
                    "scale",
                    scale_name,
                    &scale_shape,
                    &input_shape,
                    &mut nodes,
                    &mut initializers,
                );
                zero_point_name = align_param_with_input(
                    &op_name,
                    "zero_point",
                    zero_point_name,
                    &zero_point_shape,
                    &input_shape,
                    &mut nodes,
                    &mut initializers,
                );

                let input_for_div = if input_operand.descriptor.data_type == DataType::Float16 {
                    let cast_name = format!("{}_input_f32", op_name);
                    nodes.push(NodeProto {
                        input: vec![input_name.clone()],
                        output: vec![cast_name.clone()],
                        name: format!("{}_cast_input_f32", op_name),
                        op_type: "Cast".to_string(),
                        attribute: vec![AttributeProto {
                            name: "to".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: ProtoDataType::Float as i64,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    cast_name
                } else {
                    input_name.clone()
                };

                let scale_for_div = if scale_operand.descriptor.data_type == DataType::Float16 {
                    let cast_name = format!("{}_scale_f32", op_name);
                    nodes.push(NodeProto {
                        input: vec![scale_name],
                        output: vec![cast_name.clone()],
                        name: format!("{}_cast_scale_f32", op_name),
                        op_type: "Cast".to_string(),
                        attribute: vec![AttributeProto {
                            name: "to".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: ProtoDataType::Float as i64,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    cast_name
                } else {
                    scale_name
                };

                // y = clamp(roundToNearestEven(input / scale) + zeroPoint, qmin, qmax)
                let div_name = format!("{}_div", op_name);
                nodes.push(NodeProto {
                    input: vec![input_for_div, scale_for_div],
                    output: vec![div_name.clone()],
                    name: format!("{}_divide_scale", op_name),
                    op_type: "Div".to_string(),
                    ..Default::default()
                });

                let round_name = format!("{}_round", op_name);
                nodes.push(NodeProto {
                    input: vec![div_name],
                    output: vec![round_name.clone()],
                    name: format!("{}_round_even", op_name),
                    op_type: "Round".to_string(),
                    ..Default::default()
                });

                let rounded_i32_name = format!("{}_rounded_i32", op_name);
                nodes.push(NodeProto {
                    input: vec![round_name],
                    output: vec![rounded_i32_name.clone()],
                    name: format!("{}_cast_rounded_i32", op_name),
                    op_type: "Cast".to_string(),
                    attribute: vec![AttributeProto {
                        name: "to".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: ProtoDataType::Int32 as i64,
                        ..Default::default()
                    }],
                    ..Default::default()
                });

                let zp_i32_name = format!("{}_zero_point_i32", op_name);
                nodes.push(NodeProto {
                    input: vec![zero_point_name],
                    output: vec![zp_i32_name.clone()],
                    name: format!("{}_cast_zero_point_i32", op_name),
                    op_type: "Cast".to_string(),
                    attribute: vec![AttributeProto {
                        name: "to".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: ProtoDataType::Int32 as i64,
                        ..Default::default()
                    }],
                    ..Default::default()
                });

                let shifted_name = format!("{}_shifted", op_name);
                nodes.push(NodeProto {
                    input: vec![rounded_i32_name, zp_i32_name],
                    output: vec![shifted_name.clone()],
                    name: format!("{}_add_zero_point", op_name),
                    op_type: "Add".to_string(),
                    ..Default::default()
                });

                let output_operand = graph.operand(output_id).ok_or_else(|| {
                    Self::invalid_operand(
                        "quantizeLinear output lookup",
                        output_id,
                        Some((op, idx)),
                    )
                })?;
                let output_dtype = output_operand.descriptor.data_type;
                let mut quantized_i32_name = shifted_name;
                let clip_bounds = match output_dtype {
                    DataType::Int8 => Some((-128_i32, 127_i32)),
                    DataType::Uint8 => Some((0_i32, 255_i32)),
                    DataType::Int4 => Some((-8_i32, 7_i32)),
                    DataType::Uint4 => Some((0_i32, 15_i32)),
                    DataType::Int32 => None,
                    _ => {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!(
                                "Unsupported quantizeLinear output data type {:?}",
                                output_dtype
                            ),
                        });
                    }
                };

                if let Some((qmin, qmax)) = clip_bounds {
                    let qmin_name = format!("{}_qmin", op_name);
                    initializers.push(TensorProto {
                        name: qmin_name.clone(),
                        data_type: ProtoDataType::Int32 as i32,
                        dims: vec![],
                        int32_data: vec![qmin],
                        ..Default::default()
                    });
                    let qmax_name = format!("{}_qmax", op_name);
                    initializers.push(TensorProto {
                        name: qmax_name.clone(),
                        data_type: ProtoDataType::Int32 as i32,
                        dims: vec![],
                        int32_data: vec![qmax],
                        ..Default::default()
                    });

                    let clamped_low_name = format!("{}_clamped_low", op_name);
                    nodes.push(NodeProto {
                        input: vec![quantized_i32_name, qmin_name],
                        output: vec![clamped_low_name.clone()],
                        name: format!("{}_clamp_low", op_name),
                        op_type: "Max".to_string(),
                        ..Default::default()
                    });
                    let clamped_name = format!("{}_clamped", op_name);
                    nodes.push(NodeProto {
                        input: vec![clamped_low_name, qmax_name],
                        output: vec![clamped_name.clone()],
                        name: format!("{}_clamp_high", op_name),
                        op_type: "Min".to_string(),
                        ..Default::default()
                    });
                    quantized_i32_name = clamped_name;
                }

                let output_name = operand_name(graph, output_id);
                let onnx_output_dtype = Self::data_type_code(output_dtype);
                if onnx_output_dtype == ProtoDataType::Int32 {
                    nodes.push(NodeProto {
                        input: vec![quantized_i32_name],
                        output: vec![output_name],
                        name: format!("{}_output_identity", op_name),
                        op_type: "Identity".to_string(),
                        ..Default::default()
                    });
                } else {
                    nodes.push(NodeProto {
                        input: vec![quantized_i32_name],
                        output: vec![output_name],
                        name: op_name,
                        op_type: "Cast".to_string(),
                        attribute: vec![AttributeProto {
                            name: "to".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: onnx_output_dtype as i64,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                }

                continue;
            }

            if op.op_type.eq_ignore_ascii_case("dequantizeLinear") {
                let input_id = op.input_operands[0];
                let scale_id = op.input_operands[1];
                let zero_point_id = op.input_operands.get(2).copied();
                let output_id = op
                    .output_operand
                    .ok_or(GraphError::InvalidConversionOperand { operand: 0 })?;

                let input_name = operand_name(graph, input_id);
                let mut scale_name = operand_name(graph, scale_id);
                let output_name = operand_name(graph, output_id);
                let input_shape = operand_shapes.get(&input_id).cloned().unwrap_or_else(|| {
                    graph
                        .operand(input_id)
                        .map(|o| o.descriptor.static_or_max_shape())
                        .unwrap_or_default()
                });
                let scale_shape = operand_shapes.get(&scale_id).cloned().unwrap_or_else(|| {
                    graph
                        .operand(scale_id)
                        .map(|o| o.descriptor.static_or_max_shape())
                        .unwrap_or_default()
                });

                let scale_operand = graph.operand(scale_id).ok_or_else(|| {
                    Self::invalid_operand(
                        "dequantizeLinear scale lookup",
                        scale_id,
                        Some((op, idx)),
                    )
                })?;
                let output_dtype = Self::data_type_code(scale_operand.descriptor.data_type);

                fn align_param_with_input(
                    op_name: &str,
                    prefix: &str,
                    name: String,
                    param_shape: &[u32],
                    input_shape: &[u32],
                    nodes: &mut Vec<NodeProto>,
                    initializers: &mut Vec<TensorProto>,
                ) -> String {
                    if input_shape.is_empty()
                        || param_shape.is_empty()
                        || param_shape == input_shape
                        || param_shape.len() != input_shape.len()
                    {
                        return name;
                    }

                    let broadcastable = param_shape
                        .iter()
                        .zip(input_shape.iter())
                        .all(|(&p, &i)| p == 1 || p == i);
                    if broadcastable {
                        return name;
                    }

                    let tileable = param_shape
                        .iter()
                        .zip(input_shape.iter())
                        .all(|(&p, &i)| p > 0 && i % p == 0);
                    if !tileable {
                        return name;
                    }

                    let repeats: Vec<i64> = input_shape
                        .iter()
                        .zip(param_shape.iter())
                        .map(|(&i, &p)| (i / p) as i64)
                        .collect();
                    if repeats.iter().all(|&r| r == 1) {
                        return name;
                    }

                    let expanded_shape: Vec<i64> = param_shape
                        .iter()
                        .flat_map(|&d| [d as i64, 1_i64])
                        .collect();
                    let expanded_shape_name = format!("{}_{}_expand_shape", op_name, prefix);
                    initializers.push(TensorProto {
                        name: expanded_shape_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![expanded_shape.len() as i64],
                        int64_data: expanded_shape,
                        ..Default::default()
                    });

                    let expanded_name = format!("{}_{}_expanded", op_name, prefix);
                    nodes.push(NodeProto {
                        input: vec![name, expanded_shape_name],
                        output: vec![expanded_name.clone()],
                        name: format!("{}_reshape_expand_{}", op_name, prefix),
                        op_type: "Reshape".to_string(),
                        ..Default::default()
                    });

                    let tile_repeats: Vec<i64> = repeats.iter().flat_map(|&r| [1_i64, r]).collect();
                    let tile_repeats_name = format!("{}_{}_tile_repeats", op_name, prefix);
                    initializers.push(TensorProto {
                        name: tile_repeats_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![tile_repeats.len() as i64],
                        int64_data: tile_repeats,
                        ..Default::default()
                    });

                    let tiled_name = format!("{}_{}_tiled", op_name, prefix);
                    nodes.push(NodeProto {
                        input: vec![expanded_name, tile_repeats_name],
                        output: vec![tiled_name.clone()],
                        name: format!("{}_tile_{}", op_name, prefix),
                        op_type: "Tile".to_string(),
                        ..Default::default()
                    });

                    let final_shape: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
                    let final_shape_name = format!("{}_{}_final_shape", op_name, prefix);
                    initializers.push(TensorProto {
                        name: final_shape_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![final_shape.len() as i64],
                        int64_data: final_shape,
                        ..Default::default()
                    });

                    let aligned_name = format!("{}_{}_aligned", op_name, prefix);
                    nodes.push(NodeProto {
                        input: vec![tiled_name, final_shape_name],
                        output: vec![aligned_name.clone()],
                        name: format!("{}_reshape_final_{}", op_name, prefix),
                        op_type: "Reshape".to_string(),
                        ..Default::default()
                    });
                    aligned_name
                }

                scale_name = align_param_with_input(
                    &op_name,
                    "scale",
                    scale_name,
                    &scale_shape,
                    &input_shape,
                    &mut nodes,
                    &mut initializers,
                );

                let input_cast_name = format!("{}_input_cast", op_name);
                nodes.push(NodeProto {
                    input: vec![input_name],
                    output: vec![input_cast_name.clone()],
                    name: format!("{}_cast_input", op_name),
                    op_type: "Cast".to_string(),
                    attribute: vec![AttributeProto {
                        name: "to".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: output_dtype as i64,
                        ..Default::default()
                    }],
                    ..Default::default()
                });

                let centered_name = if let Some(zp_id) = zero_point_id {
                    let zp_shape = operand_shapes.get(&zp_id).cloned().unwrap_or_else(|| {
                        graph
                            .operand(zp_id)
                            .map(|o| o.descriptor.static_or_max_shape())
                            .unwrap_or_default()
                    });
                    let zp_name = align_param_with_input(
                        &op_name,
                        "zero_point",
                        operand_name(graph, zp_id),
                        &zp_shape,
                        &input_shape,
                        &mut nodes,
                        &mut initializers,
                    );
                    let zp_cast_name = format!("{}_zero_point_cast", op_name);
                    nodes.push(NodeProto {
                        input: vec![zp_name],
                        output: vec![zp_cast_name.clone()],
                        name: format!("{}_cast_zero_point", op_name),
                        op_type: "Cast".to_string(),
                        attribute: vec![AttributeProto {
                            name: "to".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: output_dtype as i64,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });

                    let sub_name = format!("{}_centered", op_name);
                    nodes.push(NodeProto {
                        input: vec![input_cast_name, zp_cast_name],
                        output: vec![sub_name.clone()],
                        name: format!("{}_subtract_zero_point", op_name),
                        op_type: "Sub".to_string(),
                        ..Default::default()
                    });
                    sub_name
                } else {
                    input_cast_name
                };

                nodes.push(NodeProto {
                    input: vec![centered_name, scale_name],
                    output: vec![output_name],
                    name: op_name,
                    op_type: "Mul".to_string(),
                    ..Default::default()
                });

                continue;
            }

            // Special-case concat: expand scalar inputs to 1D so ONNX axis validation passes
            if op.op_type.eq_ignore_ascii_case("concat") {
                let mut inputs: Vec<String> = Vec::new();

                for (input_idx, input_id) in op.input_operands.iter().enumerate() {
                    // Resolve any remapping first (for skipped concat outputs used as inputs)
                    let resolved_id = operand_remapping
                        .get(input_id)
                        .copied()
                        .unwrap_or(*input_id);

                    let operand = graph.operand(resolved_id).ok_or_else(|| {
                        debug_print!(
                            "[DEBUG] Missing operand {} in concat at op idx {}",
                            resolved_id,
                            idx
                        );
                        Self::invalid_operand("concat input lookup", resolved_id, Some((op, idx)))
                    })?;
                    // Use remapped operand name if this input was a skipped concat output
                    let input_name = {
                        let resolved_id = operand_remapping
                            .get(input_id)
                            .copied()
                            .unwrap_or(*input_id);
                        operand_name(graph, resolved_id)
                    };

                    // Use tracked shape if available, otherwise fall back to descriptor
                    // Check both original and resolved IDs for shape tracking
                    let input_shape = operand_shapes
                        .get(&resolved_id)
                        .or_else(|| operand_shapes.get(input_id))
                        .cloned()
                        .unwrap_or_else(|| operand.descriptor.static_or_max_shape());

                    if input_shape.is_empty() {
                        if let Some(data) = graph.constant_operand_ids_to_handles.get(&resolved_id)
                        {
                            // Expand scalar constant to shape [1]
                            let expanded_name = format!("{}_scalar{}_expanded", op_name, input_idx);
                            initializers.push(TensorProto {
                                name: expanded_name.clone(),
                                data_type: Self::data_type_code(operand.descriptor.data_type)
                                    as i32,
                                dims: vec![1],
                                raw_data: data.data.clone(),
                                ..Default::default()
                            });
                            inputs.push(expanded_name);
                            continue;
                        } else {
                            // Try cloning an existing initializer with the same name
                            let expanded_name = format!("{}_scalar{}_expanded", op_name, input_idx);
                            if let Some(cloned) = initializers
                                .iter()
                                .find(|t| t.name == input_name)
                                .map(|orig| {
                                    let mut cloned = orig.clone();
                                    cloned.name = expanded_name.clone();
                                    cloned.dims = vec![1];
                                    cloned
                                })
                            {
                                initializers.push(cloned);
                                inputs.push(expanded_name);
                                continue;
                            }
                        }

                        // Unknown rank: keep input unchanged and let ONNX infer rank.
                        // Inserting Unsqueeze here can over-lift rank-1 tensors to rank-2.
                        inputs.push(input_name);
                        continue;
                    }

                    inputs.push(input_name);
                }

                let attributes = Self::create_operation_attributes(op);

                // Debug: trace concat operations to find rank mismatches
                if op_name.contains("concat") {
                    debug_print!(
                        "[RUST DEBUG] Concat {} has {} inputs:",
                        op_name,
                        op.input_operands.len()
                    );
                    for (input_idx, input_id) in op.input_operands.iter().enumerate() {
                        if let Some(operand) = graph.operand(*input_id) {
                            debug_print!(
                                "  Input {}: operand_{} shape={:?} rank={}",
                                input_idx,
                                input_id,
                                operand.descriptor.shape,
                                operand.descriptor.shape.len()
                            );
                        }
                    }
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: attributes,
                    ..Default::default()
                });

                continue;
            }

            // Check if this is a logic operation that needs type conversion
            let is_not_equal_op = op.op_type.eq_ignore_ascii_case("notEqual")
                || op.op_type.eq_ignore_ascii_case("notequal");
            let is_comparison_op = is_not_equal_op
                || matches!(
                    op.op_type.as_str(),
                    "equal" | "greater" | "greaterOrEqual" | "lesser" | "lesserOrEqual"
                );
            let is_isnan_op = op.op_type.eq_ignore_ascii_case("isnan");
            let is_isinfinite_op = op.op_type.eq_ignore_ascii_case("isinfinite");
            let is_unary_predicate_op = is_isnan_op;
            let is_logical_op = matches!(
                op.op_type.as_str(),
                "logicalNot" | "logicalAnd" | "logicalOr" | "logicalXor"
            );

            if op.op_type.eq_ignore_ascii_case("argmax")
                || op.op_type.eq_ignore_ascii_case("argmin")
            {
                // ONNX ArgMax/ArgMin produce int64. Cast if WebNN output expects another integer dtype.
                let output_id = op.output_operand.expect("Single-output operation expected");
                let output_operand = graph.operand(output_id).ok_or_else(|| {
                    Self::invalid_operand("arg reduce output lookup", output_id, Some((op, idx)))
                })?;
                let output_name = operand_name(graph, output_id);
                let tmp_output = format!("{}_i64_output", op_name);
                let attributes = Self::create_operation_attributes(op);
                let input_id = op.input_operands[0];
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("arg reduce input lookup", input_id, Some((op, idx)))
                })?;
                let input_name = operand_name(graph, input_id);
                let arg_input_name = if matches!(
                    input_operand.descriptor.data_type,
                    DataType::Uint32 | DataType::Uint64 | DataType::Float16
                ) {
                    let cast_output = format!("{}_arg_input_cast", op_name);
                    nodes.push(Self::create_cast_node(
                        &format!("{}_arg_pre_cast", op_name),
                        input_name,
                        cast_output.clone(),
                        ProtoDataType::Int64,
                    ));
                    cast_output
                } else {
                    input_name
                };

                nodes.push(NodeProto {
                    input: vec![arg_input_name],
                    output: vec![tmp_output.clone()],
                    name: op_name.clone(),
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: attributes,
                    ..Default::default()
                });

                let target_dtype = Self::data_type_code(output_operand.descriptor.data_type);
                if target_dtype == ProtoDataType::Int64 {
                    nodes.push(NodeProto {
                        input: vec![tmp_output],
                        output: vec![output_name],
                        name: format!("{}_identity_output", op_name),
                        op_type: "Identity".to_string(),
                        attribute: vec![],
                        ..Default::default()
                    });
                } else {
                    nodes.push(Self::create_cast_node(
                        &format!("{}_cast_output", op_name),
                        tmp_output,
                        output_name,
                        target_dtype,
                    ));
                }
                continue;
            }

            if op.op_type.eq_ignore_ascii_case("averagepool2d")
                || op.op_type.eq_ignore_ascii_case("maxpool2d")
            {
                let input_id = op.input_operands[0];
                let input_name = operand_name(graph, input_id);
                let output_name = operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );
                let layout = op
                    .attributes
                    .get("layout")
                    .and_then(|v| v.as_str())
                    .unwrap_or("nchw")
                    .to_ascii_lowercase();

                // ONNX AveragePool does not support dilations. Emulate dilated average pool
                // (for no-padding cases) with depthwise Conv using sparse kernel taps.
                let is_avg_pool = op.op_type.eq_ignore_ascii_case("averagepool2d");
                let dilations = Self::parse_i64_array_any(op, &["dilations", "dilation"])
                    .unwrap_or_else(|| vec![1, 1]);
                let strides = Self::parse_i64_array_any(op, &["strides", "stride"])
                    .unwrap_or_else(|| vec![1, 1]);
                let pads = Self::parse_pads_attr(op).unwrap_or_else(|| vec![0, 0, 0, 0]);
                let kernel_shape = Self::parse_i64_array_any(
                    op,
                    &[
                        "windowDimensions",
                        "window_dimensions",
                        "windowdimensions",
                        "kernelShape",
                        "kernel_shape",
                    ],
                );
                let can_emulate_dilated_avg = is_avg_pool
                    && dilations.len() == 2
                    && (dilations[0] > 1 || dilations[1] > 1)
                    && strides.len() == 2
                    && pads.len() == 4
                    && pads.iter().all(|&p| p == 0)
                    && kernel_shape
                        .as_ref()
                        .map(|k| k.len() == 2 && k[0] > 0 && k[1] > 0)
                        .unwrap_or(false);

                if can_emulate_dilated_avg {
                    let input_operand = graph.operand(input_id).ok_or_else(|| {
                        Self::invalid_operand(
                            "averagepool2d input lookup",
                            input_id,
                            Some((op, idx)),
                        )
                    })?;
                    let input_shape = input_operand.descriptor.static_or_max_shape();
                    if input_shape.len() == 4 {
                        let channels = if layout == "nhwc" {
                            input_shape[3] as i64
                        } else {
                            input_shape[1] as i64
                        };
                        if channels > 0 {
                            let k = kernel_shape.expect("validated kernel shape");
                            let kh = k[0];
                            let kw = k[1];
                            let dh = dilations[0];
                            let dw = dilations[1];
                            let eff_h = dh * (kh - 1) + 1;
                            let eff_w = dw * (kw - 1) + 1;
                            let denom = (kh * kw) as f32;

                            let mut conv_input_name = input_name.clone();
                            if layout == "nhwc" {
                                let nchw_input = format!("{}_nchw_in", op_name);
                                nodes.push(NodeProto {
                                    input: vec![input_name],
                                    output: vec![nchw_input.clone()],
                                    name: format!("{}_to_nchw", op_name),
                                    op_type: "Transpose".to_string(),
                                    attribute: vec![AttributeProto {
                                        name: "perm".to_string(),
                                        r#type: AttributeType::Ints as i32,
                                        ints: vec![0, 3, 1, 2],
                                        ..Default::default()
                                    }],
                                    ..Default::default()
                                });
                                conv_input_name = nchw_input;
                            }

                            let weight_name = format!("{}_dilated_avg_weight", op_name);
                            let kernel_elem_count = (channels * eff_h * eff_w) as usize;
                            let mut weight_vals = vec![0f32; kernel_elem_count];
                            for c in 0..(channels as usize) {
                                for ky in 0..(kh as usize) {
                                    for kx in 0..(kw as usize) {
                                        let y = ky * (dh as usize);
                                        let x = kx * (dw as usize);
                                        let idx = c * (eff_h as usize) * (eff_w as usize)
                                            + y * (eff_w as usize)
                                            + x;
                                        weight_vals[idx] = 1.0 / denom;
                                    }
                                }
                            }

                            let input_dtype =
                                Self::data_type_code(input_operand.descriptor.data_type);
                            initializers.push(TensorProto {
                                name: weight_name.clone(),
                                data_type: ProtoDataType::Float as i32,
                                dims: vec![channels, 1, eff_h, eff_w],
                                float_data: weight_vals,
                                ..Default::default()
                            });

                            let conv_output_name = if layout == "nhwc" {
                                format!("{}_nchw_out", op_name)
                            } else {
                                output_name.clone()
                            };
                            let mut conv_attributes = Vec::new();
                            Self::add_ints_attribute(&mut conv_attributes, "strides", strides);
                            Self::add_int_attribute(&mut conv_attributes, "group", channels);

                            let conv_input_f32 = if input_dtype == ProtoDataType::Float16 {
                                let casted = format!("{}_dilated_avg_input_f32", op_name);
                                nodes.push(Self::create_cast_node(
                                    &format!("{}_dilated_avg_cast_in", op_name),
                                    conv_input_name,
                                    casted.clone(),
                                    ProtoDataType::Float,
                                ));
                                casted
                            } else {
                                conv_input_name
                            };

                            let conv_output_raw = if input_dtype == ProtoDataType::Float16 {
                                format!("{}_dilated_avg_output_f32", op_name)
                            } else {
                                conv_output_name.clone()
                            };

                            nodes.push(NodeProto {
                                input: vec![conv_input_f32, weight_name],
                                output: vec![conv_output_raw.clone()],
                                name: op_name.clone(),
                                op_type: "Conv".to_string(),
                                attribute: conv_attributes,
                                ..Default::default()
                            });

                            if input_dtype == ProtoDataType::Float16 {
                                nodes.push(Self::create_cast_node(
                                    &format!("{}_dilated_avg_cast_out", op_name),
                                    conv_output_raw,
                                    conv_output_name.clone(),
                                    ProtoDataType::Float16,
                                ));
                            }

                            if layout == "nhwc" {
                                nodes.push(NodeProto {
                                    input: vec![conv_output_name],
                                    output: vec![output_name],
                                    name: format!("{}_to_nhwc", op_name),
                                    op_type: "Transpose".to_string(),
                                    attribute: vec![AttributeProto {
                                        name: "perm".to_string(),
                                        r#type: AttributeType::Ints as i32,
                                        ints: vec![0, 2, 3, 1],
                                        ..Default::default()
                                    }],
                                    ..Default::default()
                                });
                            }

                            continue;
                        }
                    }
                }

                let attributes = Self::create_pool2d_attributes_with_graph(op, graph);

                if layout == "nhwc" {
                    let nchw_input = format!("{}_nchw_in", op_name);
                    nodes.push(NodeProto {
                        input: vec![input_name],
                        output: vec![nchw_input.clone()],
                        name: format!("{}_to_nchw", op_name),
                        op_type: "Transpose".to_string(),
                        attribute: vec![AttributeProto {
                            name: "perm".to_string(),
                            r#type: AttributeType::Ints as i32,
                            ints: vec![0, 3, 1, 2],
                            ..Default::default()
                        }],
                        ..Default::default()
                    });

                    let nchw_output = format!("{}_nchw_out", op_name);
                    nodes.push(NodeProto {
                        input: vec![nchw_input],
                        output: vec![nchw_output.clone()],
                        name: op_name.clone(),
                        op_type: Self::onnx_op_type(&op.op_type),
                        attribute: attributes,
                        ..Default::default()
                    });

                    nodes.push(NodeProto {
                        input: vec![nchw_output],
                        output: vec![output_name],
                        name: format!("{}_to_nhwc", op_name),
                        op_type: "Transpose".to_string(),
                        attribute: vec![AttributeProto {
                            name: "perm".to_string(),
                            r#type: AttributeType::Ints as i32,
                            ints: vec![0, 2, 3, 1],
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                } else {
                    nodes.push(NodeProto {
                        input: vec![input_name],
                        output: vec![output_name],
                        name: op_name.clone(),
                        op_type: Self::onnx_op_type(&op.op_type),
                        attribute: attributes,
                        ..Default::default()
                    });
                }
                continue;
            }

            if is_logical_op {
                // Logical operations: Cast inputs to bool, execute op, cast output to uint8
                let mut cast_inputs = Vec::new();

                for &input_id in &op.input_operands {
                    let input_name = operand_name(graph, input_id);
                    let cast_output_name = format!("cast_to_bool_{}_{}", op_name, cast_counter);
                    cast_counter += 1;

                    // Create Cast node: input type -> bool
                    nodes.push(Self::create_cast_node(
                        &format!("cast_to_bool_{}", cast_counter - 1),
                        input_name,
                        cast_output_name.clone(),
                        ProtoDataType::Bool,
                    ));

                    cast_inputs.push(cast_output_name);
                }

                // Create the logical operation node (outputs bool)
                let bool_output_name = format!("{}_bool_output", op_name);
                let attributes = Self::create_operation_attributes(op);

                nodes.push(NodeProto {
                    input: cast_inputs,
                    output: vec![bool_output_name.clone()],
                    name: op_name.clone(),
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: attributes,
                    ..Default::default()
                });

                // Cast bool → uint8 (matching Chromium's WebNN implementation)
                nodes.push(Self::create_cast_node(
                    &format!("cast_to_uint8_{}", cast_counter),
                    bool_output_name,
                    operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    ),
                    ProtoDataType::Uint8,
                ));
                cast_counter += 1;
            } else if is_comparison_op {
                // Comparison operations: execute op (or decomposition), then cast bool output to uint8
                let bool_output_name = format!("{}_bool_output", op_name);
                let attributes = Self::create_operation_attributes(op);
                let output_operand_id =
                    op.output_operand.expect("Single-output operation expected");
                let output_name = operand_name(graph, output_operand_id);
                let input_names: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                if is_not_equal_op {
                    // ONNX has no NotEqual op; lower to Not(Equal(x, y)).
                    let equal_output_name = format!("{}_equal_output", op_name);
                    nodes.push(NodeProto {
                        input: input_names,
                        output: vec![equal_output_name.clone()],
                        name: format!("{}_equal", op_name),
                        op_type: "Equal".to_string(),
                        attribute: vec![],
                        ..Default::default()
                    });
                    nodes.push(NodeProto {
                        input: vec![equal_output_name],
                        output: vec![bool_output_name.clone()],
                        name: format!("{}_not", op_name),
                        op_type: "Not".to_string(),
                        attribute: vec![],
                        ..Default::default()
                    });
                } else {
                    // Create comparison node (outputs bool)
                    nodes.push(NodeProto {
                        input: input_names,
                        output: vec![bool_output_name.clone()],
                        name: op_name,
                        op_type: Self::onnx_op_type(&op.op_type),
                        attribute: attributes,
                        ..Default::default()
                    });
                }

                // Cast bool → uint8 (matching Chromium's WebNN implementation)
                nodes.push(Self::create_cast_node(
                    &format!("cast_to_uint8_{}", cast_counter),
                    bool_output_name,
                    output_name,
                    ProtoDataType::Uint8,
                ));
                cast_counter += 1;
            } else if is_isinfinite_op {
                // isInfinite: lower to Equal(Abs(x), +inf) for ORT compatibility.
                let input_id = op.input_operands[0];
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("isInfinite input", input_id, Some((op, idx)))
                })?;
                let input_name = operand_name(graph, input_id);

                let abs_output = format!("{}_abs_output", op_name);
                nodes.push(NodeProto {
                    input: vec![input_name],
                    output: vec![abs_output.clone()],
                    name: format!("{}_abs", op_name),
                    op_type: "Abs".to_string(),
                    attribute: vec![],
                    ..Default::default()
                });

                let inf_name = format!("{}_inf_const", op_name);
                let inf_dtype = Self::data_type_code(input_operand.descriptor.data_type);
                initializers.push(Self::create_scalar_initializer(
                    inf_name.clone(),
                    inf_dtype,
                    f32::INFINITY,
                ));

                let bool_output_name = format!("{}_bool_output", op_name);
                nodes.push(NodeProto {
                    input: vec![abs_output, inf_name],
                    output: vec![bool_output_name.clone()],
                    name: format!("{}_equal_inf", op_name),
                    op_type: "Equal".to_string(),
                    attribute: vec![],
                    ..Default::default()
                });

                nodes.push(Self::create_cast_node(
                    &format!("cast_to_uint8_{}", cast_counter),
                    bool_output_name,
                    operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    ),
                    ProtoDataType::Uint8,
                ));
                cast_counter += 1;
            } else if is_unary_predicate_op {
                // Unary predicate (isNaN): execute op (bool), cast output to uint8.
                let bool_output_name = format!("{}_bool_output", op_name);

                nodes.push(NodeProto {
                    input: op
                        .input_operands
                        .iter()
                        .map(|id| operand_name(graph, *id))
                        .collect(),
                    output: vec![bool_output_name.clone()],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: vec![],
                    ..Default::default()
                });

                nodes.push(Self::create_cast_node(
                    &format!("cast_to_uint8_{}", cast_counter),
                    bool_output_name,
                    operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    ),
                    ProtoDataType::Uint8,
                ));
                cast_counter += 1;
            } else if op.op_type.eq_ignore_ascii_case("gelu") {
                // GELU exact formulation:
                // 0.5 * x * (1 + erf(x / sqrt(2)))
                // Avoid ONNX Gelu op because it is unavailable in our current ORT build.
                let input_id = op.input_operands[0];
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("gelu input", input_id, Some((op, idx)))
                })?;
                let input_name = operand_name(graph, input_id);
                let output_name = operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );

                let dtype = input_operand.descriptor.data_type;
                let out_dtype = Self::data_type_code(dtype);
                let (compute_input, needs_cast_back) = if dtype == DataType::Float16 {
                    let cast_name = format!("{}_gelu_input_f32", op_name);
                    nodes.push(Self::create_cast_node(
                        &format!("{}_gelu_cast_input", op_name),
                        input_name,
                        cast_name.clone(),
                        ProtoDataType::Float,
                    ));
                    (cast_name, true)
                } else {
                    (input_name, false)
                };

                let half_name = format!("{}_gelu_half", op_name);
                let one_name = format!("{}_gelu_one", op_name);
                let inv_sqrt2_name = format!("{}_gelu_inv_sqrt2", op_name);
                initializers.push(Self::create_scalar_initializer(
                    half_name.clone(),
                    ProtoDataType::Float,
                    0.5,
                ));
                initializers.push(Self::create_scalar_initializer(
                    one_name.clone(),
                    ProtoDataType::Float,
                    1.0,
                ));
                initializers.push(Self::create_scalar_initializer(
                    inv_sqrt2_name.clone(),
                    ProtoDataType::Float,
                    std::f32::consts::FRAC_1_SQRT_2,
                ));

                let x_div_name = format!("{}_gelu_x_div_sqrt2", op_name);
                nodes.push(NodeProto {
                    input: vec![compute_input.clone(), inv_sqrt2_name],
                    output: vec![x_div_name.clone()],
                    name: format!("{}_gelu_div", op_name),
                    op_type: "Mul".to_string(),
                    ..Default::default()
                });
                let erf_name = format!("{}_gelu_erf", op_name);
                nodes.push(NodeProto {
                    input: vec![x_div_name],
                    output: vec![erf_name.clone()],
                    name: format!("{}_gelu_erf", op_name),
                    op_type: "Erf".to_string(),
                    ..Default::default()
                });
                let one_plus_erf = format!("{}_gelu_one_plus_erf", op_name);
                nodes.push(NodeProto {
                    input: vec![one_name, erf_name],
                    output: vec![one_plus_erf.clone()],
                    name: format!("{}_gelu_add", op_name),
                    op_type: "Add".to_string(),
                    ..Default::default()
                });
                let x_times_term = format!("{}_gelu_x_times_term", op_name);
                nodes.push(NodeProto {
                    input: vec![compute_input, one_plus_erf],
                    output: vec![x_times_term.clone()],
                    name: format!("{}_gelu_mul1", op_name),
                    op_type: "Mul".to_string(),
                    ..Default::default()
                });
                let final_compute_name = if needs_cast_back {
                    format!("{}_gelu_out_f32", op_name)
                } else {
                    output_name.clone()
                };
                nodes.push(NodeProto {
                    input: vec![x_times_term, half_name],
                    output: vec![final_compute_name.clone()],
                    name: format!("{}_gelu_mul2", op_name),
                    op_type: "Mul".to_string(),
                    ..Default::default()
                });

                if needs_cast_back {
                    nodes.push(Self::create_cast_node(
                        &format!("{}_gelu_cast_output", op_name),
                        final_compute_name,
                        output_name,
                        out_dtype,
                    ));
                }
            } else if op.op_type.eq_ignore_ascii_case("linear") {
                // linear: y = alpha * x + beta
                // Lower to primitive Mul + Add for broad ONNX Runtime compatibility.
                let input_id = op.input_operands[0];
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("linear input", input_id, Some((op, idx)))
                })?;
                if !matches!(
                    input_operand.descriptor.data_type,
                    DataType::Float32 | DataType::Float16
                ) {
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!(
                            "linear currently supports float32/float16, got {:?}",
                            input_operand.descriptor.data_type
                        ),
                    });
                }
                let input_name = operand_name(graph, input_id);
                let output_name = operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );

                let alpha = Self::parse_f64_attr(op.attributes.get("alpha")).unwrap_or(1.0) as f32;
                let beta = Self::parse_f64_attr(op.attributes.get("beta")).unwrap_or(0.0) as f32;

                // Compute in float32 for float16 inputs, then cast back.
                let (compute_input, needs_cast_back) =
                    if input_operand.descriptor.data_type == DataType::Float16 {
                        let cast_name = format!("{}_input_f32", op_name);
                        nodes.push(Self::create_cast_node(
                            &format!("{}_cast_input", op_name),
                            input_name,
                            cast_name.clone(),
                            ProtoDataType::Float,
                        ));
                        (cast_name, true)
                    } else {
                        (input_name, false)
                    };

                let alpha_name = format!("{}_linear_alpha", op_name);
                let beta_name = format!("{}_linear_beta", op_name);
                initializers.push(TensorProto {
                    name: alpha_name.clone(),
                    data_type: ProtoDataType::Float as i32,
                    dims: vec![],
                    float_data: vec![alpha],
                    ..Default::default()
                });
                initializers.push(TensorProto {
                    name: beta_name.clone(),
                    data_type: ProtoDataType::Float as i32,
                    dims: vec![],
                    float_data: vec![beta],
                    ..Default::default()
                });

                let scaled_name = format!("{}_linear_scaled", op_name);
                let final_compute_name = if needs_cast_back {
                    format!("{}_linear_out_f32", op_name)
                } else {
                    output_name.clone()
                };

                nodes.push(NodeProto {
                    input: vec![compute_input, alpha_name],
                    output: vec![scaled_name.clone()],
                    name: format!("{}_linear_mul", op_name),
                    op_type: "Mul".to_string(),
                    ..Default::default()
                });
                nodes.push(NodeProto {
                    input: vec![scaled_name, beta_name],
                    output: vec![final_compute_name.clone()],
                    name: format!("{}_linear_add", op_name),
                    op_type: "Add".to_string(),
                    ..Default::default()
                });

                if needs_cast_back {
                    nodes.push(Self::create_cast_node(
                        &format!("{}_cast_output", op_name),
                        final_compute_name,
                        output_name,
                        ProtoDataType::Float16,
                    ));
                }
            } else if op.op_type.eq_ignore_ascii_case("triangular") {
                // Triangular operation: Cast integer inputs to float32
                let input_id = op.input_operands[0];
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("triangular input", input_id, Some((op, idx)))
                })?;

                let needs_cast = matches!(
                    input_operand.descriptor.data_type,
                    DataType::Int32 | DataType::Int64 | DataType::Uint32 | DataType::Uint8
                );

                let input_name = if needs_cast {
                    let cast_output = format!("{}_input_float", op_name);
                    debug_print!(
                        "[FIX] Triangular op {} has {:?} input, casting to Float32",
                        op_name,
                        input_operand.descriptor.data_type
                    );
                    nodes.push(Self::create_cast_node(
                        &format!("{}_pre_cast", op_name),
                        operand_name(graph, input_id),
                        cast_output.clone(),
                        ProtoDataType::Float,
                    ));
                    cast_output
                } else {
                    operand_name(graph, input_id)
                };

                let mut inputs = vec![input_name];
                let mut attributes = Vec::new();
                if let Some(upper) = op.attributes.get("upper").and_then(|v| v.as_bool()) {
                    attributes.push(AttributeProto {
                        name: "upper".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: if upper { 1 } else { 0 },
                        ..Default::default()
                    });
                }
                if let Some(k) = op
                    .attributes
                    .get("diagonal")
                    .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
                {
                    let k_name = format!("{}_k", op_name);
                    initializers.push(TensorProto {
                        name: k_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![],
                        int64_data: vec![k],
                        ..Default::default()
                    });
                    inputs.push(k_name);
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name.clone(),
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if op.op_type.eq_ignore_ascii_case("where") {
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Ensure condition is bool for ONNX Where (WebNN uses uint8 for comparisons)
                if !inputs.is_empty() {
                    let cast_name = format!("{}_cond_bool", op_name);
                    nodes.push(Self::create_cast_node(
                        &cast_name,
                        inputs[0].clone(),
                        cast_name.clone(),
                        ProtoDataType::Bool,
                    ));
                    inputs[0] = cast_name;
                }

                // Ensure both value inputs have identical dtype for ONNX Where.
                if op.input_operands.len() >= 3 {
                    let true_id = op.input_operands[1];
                    let false_id = op.input_operands[2];
                    let target_type = graph
                        .operand(true_id)
                        .map(|operand| {
                            type_overrides
                                .get(&true_id)
                                .copied()
                                .unwrap_or(operand.descriptor.data_type)
                        })
                        .ok_or_else(|| {
                            Self::invalid_operand("where true input", true_id, Some((op, idx)))
                        })?;

                    let true_cast_name = format!("{}_true_cast_{}", op_name, cast_counter);
                    cast_counter += 1;
                    nodes.push(Self::create_cast_node(
                        &format!("{}_cast_true_{}", op_name, cast_counter),
                        inputs[1].clone(),
                        true_cast_name.clone(),
                        Self::data_type_code(target_type),
                    ));
                    inputs[1] = true_cast_name;

                    // Validate false operand exists for clearer converter errors.
                    if graph.operand(false_id).is_none() {
                        return Err(Self::invalid_operand(
                            "where false input",
                            false_id,
                            Some((op, idx)),
                        ));
                    }

                    let false_cast_name = format!("{}_false_cast_{}", op_name, cast_counter);
                    cast_counter += 1;
                    nodes.push(Self::create_cast_node(
                        &format!("{}_cast_false_{}", op_name, cast_counter),
                        inputs[2].clone(),
                        false_cast_name.clone(),
                        Self::data_type_code(target_type),
                    ));
                    inputs[2] = false_cast_name;

                    if let Some(output_id) = op.output_operand {
                        type_overrides.insert(output_id, target_type);
                    }
                }

                let attributes = Self::create_operation_attributes(op);

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if op.op_type.eq_ignore_ascii_case("reverse") {
                // ONNX has no standard "Reverse" op; lower WebNN reverse to a sequence of Slice
                // ops with step=-1 for each target axis.
                let input_id = *op.input_operands.first().ok_or_else(|| {
                    Self::invalid_operand("reverse missing data input", idx as u32, Some((op, idx)))
                })?;
                let input_name = operand_name(graph, input_id);
                let output_id = op.output_operand.expect("Single-output operation expected");
                let final_output_name = operand_name(graph, output_id);

                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("reverse data input lookup", input_id, Some((op, idx)))
                })?;
                let rank = input_operand.descriptor.shape.len();
                let rank_i64 = rank as i64;

                let axes = if let Some(axes_val) = op.attributes.get("axes") {
                    let arr = axes_val
                        .as_array()
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: "reverse axes must be an array".to_string(),
                        })?;
                    let mut parsed = Vec::with_capacity(arr.len());
                    for value in arr {
                        let axis = value.as_i64().ok_or_else(|| GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!("reverse axis must be integer, got {value}"),
                        })?;
                        parsed.push(axis);
                    }
                    parsed
                } else {
                    (0..rank_i64).collect()
                };

                let mut normalized_axes = Vec::with_capacity(axes.len());
                for axis in axes {
                    let mut axis = axis;
                    if axis < 0 {
                        axis += rank_i64;
                    }
                    if axis < 0 || axis >= rank_i64 {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!("reverse axis {axis} out of range for rank {rank}"),
                        });
                    }
                    normalized_axes.push(axis);
                }
                normalized_axes.sort_unstable();
                normalized_axes.dedup();

                if normalized_axes.is_empty() {
                    nodes.push(NodeProto {
                        input: vec![input_name],
                        output: vec![final_output_name.clone()],
                        name: format!("{}_identity", op_name),
                        op_type: "Identity".to_string(),
                        attribute: vec![],
                        ..Default::default()
                    });
                    type_overrides.insert(output_id, input_operand.descriptor.data_type);
                    continue;
                }

                let mut current_input = input_name;
                for (axis_index, axis) in normalized_axes.iter().enumerate() {
                    let starts_name = format!("{}_reverse_{}_starts", op_name, axis_index);
                    let ends_name = format!("{}_reverse_{}_ends", op_name, axis_index);
                    let axes_name = format!("{}_reverse_{}_axes", op_name, axis_index);
                    let steps_name = format!("{}_reverse_{}_steps", op_name, axis_index);

                    initializers.push(TensorProto {
                        name: starts_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![1],
                        int64_data: vec![-1],
                        ..Default::default()
                    });
                    initializers.push(TensorProto {
                        name: ends_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![1],
                        int64_data: vec![i64::MIN],
                        ..Default::default()
                    });
                    initializers.push(TensorProto {
                        name: axes_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![1],
                        int64_data: vec![*axis],
                        ..Default::default()
                    });
                    initializers.push(TensorProto {
                        name: steps_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![1],
                        int64_data: vec![-1],
                        ..Default::default()
                    });

                    let slice_output = if axis_index + 1 == normalized_axes.len() {
                        final_output_name.clone()
                    } else {
                        format!("{}_reverse_{}_out", op_name, axis_index)
                    };

                    nodes.push(NodeProto {
                        input: vec![
                            current_input.clone(),
                            starts_name,
                            ends_name,
                            axes_name,
                            steps_name,
                        ],
                        output: vec![slice_output.clone()],
                        name: format!("{}_reverse_slice_{}", op_name, axis_index),
                        op_type: "Slice".to_string(),
                        attribute: vec![],
                        ..Default::default()
                    });
                    current_input = slice_output;
                }
                type_overrides.insert(output_id, input_operand.descriptor.data_type);
            } else if op.op_type.eq_ignore_ascii_case("cumulativeSum")
                || op.op_type.eq_ignore_ascii_case("cumulative_sum")
            {
                // ONNX CumSum requires axis as second input tensor.
                let input_id = *op.input_operands.first().ok_or_else(|| {
                    Self::invalid_operand(
                        "cumulativeSum missing data input",
                        idx as u32,
                        Some((op, idx)),
                    )
                })?;
                let input_name = operand_name(graph, input_id);
                let output_id = op.output_operand.expect("Single-output operation expected");
                let output_name = operand_name(graph, output_id);

                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("cumulativeSum input lookup", input_id, Some((op, idx)))
                })?;
                let rank = input_operand.descriptor.shape.len() as i64;

                let mut axis = op
                    .attributes
                    .get("axis")
                    .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
                    .unwrap_or(0);
                if rank > 0 && axis < 0 {
                    axis += rank;
                }
                if rank > 0 && (axis < 0 || axis >= rank) {
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!(
                            "cumulativeSum axis {} out of bounds for rank {}",
                            axis, rank
                        ),
                    });
                }

                let axis_name = format!("{}_axis", op_name);
                initializers.push(TensorProto {
                    name: axis_name.clone(),
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![1],
                    int64_data: vec![axis],
                    ..Default::default()
                });

                let exclusive = op
                    .attributes
                    .get("exclusive")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reverse = op
                    .attributes
                    .get("reverse")
                    .or_else(|| op.attributes.get("reversed"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let mut attributes = Vec::new();
                if exclusive {
                    attributes.push(AttributeProto {
                        name: "exclusive".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: 1,
                        ..Default::default()
                    });
                }
                if reverse {
                    attributes.push(AttributeProto {
                        name: "reverse".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: 1,
                        ..Default::default()
                    });
                }

                nodes.push(NodeProto {
                    input: vec![input_name, axis_name],
                    output: vec![output_name],
                    name: op_name,
                    op_type: "CumSum".to_string(),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if op.op_type.eq_ignore_ascii_case("grucell")
                || op.op_type.eq_ignore_ascii_case("gru_cell")
            {
                if op.input_operands.len() < 4 {
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!(
                            "gruCell requires at least 4 inputs (input, weight, recurrentWeight, hiddenState), got {}",
                            op.input_operands.len()
                        ),
                    });
                }

                let input_id = op.input_operands[0];
                let weight_id = op.input_operands[1];
                let recurrent_weight_id = op.input_operands[2];
                let hidden_state_id = op.input_operands[3];
                let output_id = op.output_operand.expect("Single-output operation expected");

                let input_name = operand_name(graph, input_id);
                let weight_name = operand_name(graph, weight_id);
                let recurrent_weight_name = operand_name(graph, recurrent_weight_id);
                let hidden_state_name = operand_name(graph, hidden_state_id);
                let final_output_name = operand_name(graph, output_id);

                let hidden_size = op
                    .attributes
                    .get("hiddenSize")
                    .or_else(|| op.attributes.get("hidden_size"))
                    .and_then(|v| v.as_u64().or_else(|| v.as_i64().map(|x| x as u64)))
                    .or_else(|| {
                        graph.operand(output_id).and_then(|o| {
                            o.descriptor
                                .shape
                                .last()
                                .map(|d| get_static_or_max_size(d) as u64)
                        })
                    })
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: "gruCell missing hiddenSize/hidden_size attribute".to_string(),
                    })?;

                let hidden_state_operand = graph.operand(hidden_state_id).ok_or_else(|| {
                    Self::invalid_operand(
                        "gruCell hiddenState lookup",
                        hidden_state_id,
                        Some((op, idx)),
                    )
                })?;
                let input_dtype = Self::data_type_code(hidden_state_operand.descriptor.data_type);

                let gate_layout = op
                    .attributes
                    .get("layout")
                    .and_then(|v| v.as_str())
                    .unwrap_or("zrn")
                    .to_ascii_lowercase();
                let needs_rzn_to_zrn = gate_layout == "rzn";

                let axes0_name = format!("{}_axes0", op_name);
                initializers.push(TensorProto {
                    name: axes0_name.clone(),
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![1],
                    int64_data: vec![0],
                    ..Default::default()
                });

                let make_slice_const =
                    |suffix: &str, values: Vec<i64>, initializers: &mut Vec<TensorProto>| {
                        let name = format!("{}_{}", op_name, suffix);
                        initializers.push(TensorProto {
                            name: name.clone(),
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![values.len() as i64],
                            int64_data: values,
                            ..Default::default()
                        });
                        name
                    };

                let reorder_rzn_gate_chunks = |base_name: &str,
                                               tensor_name: String,
                                               nodes: &mut Vec<NodeProto>,
                                               initializers: &mut Vec<TensorProto>|
                 -> String {
                    let mut chunks = Vec::with_capacity(3);
                    for gate_idx in 0..3 {
                        let starts_name = make_slice_const(
                            &format!("{}_slice{}_starts", base_name, gate_idx),
                            vec![(gate_idx * hidden_size as usize) as i64],
                            initializers,
                        );
                        let ends_name = make_slice_const(
                            &format!("{}_slice{}_ends", base_name, gate_idx),
                            vec![((gate_idx + 1) * hidden_size as usize) as i64],
                            initializers,
                        );
                        let steps_name = make_slice_const(
                            &format!("{}_slice{}_steps", base_name, gate_idx),
                            vec![1],
                            initializers,
                        );
                        let chunk_name = format!("{}_{}_chunk{}", op_name, base_name, gate_idx);
                        nodes.push(NodeProto {
                            input: vec![
                                tensor_name.clone(),
                                starts_name,
                                ends_name,
                                axes0_name.clone(),
                                steps_name,
                            ],
                            output: vec![chunk_name.clone()],
                            name: format!("{}_{}_slice{}", op_name, base_name, gate_idx),
                            op_type: "Slice".to_string(),
                            ..Default::default()
                        });
                        chunks.push(chunk_name);
                    }

                    let reordered_name = format!("{}_{}_zrn", op_name, base_name);
                    nodes.push(NodeProto {
                        input: vec![chunks[1].clone(), chunks[0].clone(), chunks[2].clone()],
                        output: vec![reordered_name.clone()],
                        name: format!("{}_{}_reorder", op_name, base_name),
                        op_type: "Concat".to_string(),
                        attribute: vec![AttributeProto {
                            name: "axis".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: 0,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    reordered_name
                };

                let w_for_gru = if needs_rzn_to_zrn {
                    reorder_rzn_gate_chunks("w", weight_name, &mut nodes, &mut initializers)
                } else {
                    weight_name
                };
                let r_for_gru = if needs_rzn_to_zrn {
                    reorder_rzn_gate_chunks(
                        "r",
                        recurrent_weight_name,
                        &mut nodes,
                        &mut initializers,
                    )
                } else {
                    recurrent_weight_name
                };

                let mut bias_name = if op.input_operands.len() > 4 {
                    operand_name(graph, op.input_operands[4])
                } else {
                    let name = format!("{}_bias_zero", op_name);
                    initializers.push(Self::create_vector_initializer(
                        name.clone(),
                        input_dtype,
                        vec![3 * hidden_size as i64],
                        0.0,
                    ));
                    name
                };
                let mut recurrent_bias_name = if op.input_operands.len() > 5 {
                    operand_name(graph, op.input_operands[5])
                } else {
                    let name = format!("{}_recurrent_bias_zero", op_name);
                    initializers.push(Self::create_vector_initializer(
                        name.clone(),
                        input_dtype,
                        vec![3 * hidden_size as i64],
                        0.0,
                    ));
                    name
                };
                if needs_rzn_to_zrn {
                    bias_name =
                        reorder_rzn_gate_chunks("b", bias_name, &mut nodes, &mut initializers);
                    recurrent_bias_name = reorder_rzn_gate_chunks(
                        "rb",
                        recurrent_bias_name,
                        &mut nodes,
                        &mut initializers,
                    );
                }

                let combined_bias_name = format!("{}_combined_bias", op_name);
                nodes.push(NodeProto {
                    input: vec![bias_name, recurrent_bias_name],
                    output: vec![combined_bias_name.clone()],
                    name: format!("{}_combine_biases", op_name),
                    op_type: "Concat".to_string(),
                    attribute: vec![AttributeProto {
                        name: "axis".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: 0,
                        ..Default::default()
                    }],
                    ..Default::default()
                });

                let x_seq_name = format!("{}_x_seq", op_name);
                let w_3d_name = format!("{}_w_3d", op_name);
                let r_3d_name = format!("{}_r_3d", op_name);
                let b_2d_name = format!("{}_b_2d", op_name);
                let h_3d_name = format!("{}_h_3d", op_name);

                for (src, dst, label) in [
                    (input_name, x_seq_name.clone(), "x"),
                    (w_for_gru, w_3d_name.clone(), "w"),
                    (r_for_gru, r_3d_name.clone(), "r"),
                    (combined_bias_name, b_2d_name.clone(), "b"),
                    (hidden_state_name, h_3d_name.clone(), "h"),
                ] {
                    nodes.push(NodeProto {
                        input: vec![src, axes0_name.clone()],
                        output: vec![dst],
                        name: format!("{}_unsqueeze_{}", op_name, label),
                        op_type: "Unsqueeze".to_string(),
                        ..Default::default()
                    });
                }

                let gru_y_name = format!("{}_y", op_name);
                let gru_y_h_name = format!("{}_y_h", op_name);
                let reset_after = op
                    .attributes
                    .get("resetAfter")
                    .or_else(|| op.attributes.get("reset_after"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let mut gru_attrs = vec![
                    AttributeProto {
                        name: "hidden_size".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: hidden_size as i64,
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "linear_before_reset".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: if reset_after { 1 } else { 0 },
                        ..Default::default()
                    },
                ];

                if let Some(activations) =
                    op.attributes.get("activations").and_then(|v| v.as_array())
                {
                    let strings: Vec<Vec<u8>> = activations
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .map(|s| s.into_bytes())
                        .collect();
                    if !strings.is_empty() {
                        gru_attrs.push(AttributeProto {
                            name: "activations".to_string(),
                            r#type: AttributeType::Strings as i32,
                            strings,
                            ..Default::default()
                        });
                    }
                }

                nodes.push(NodeProto {
                    input: vec![
                        x_seq_name,
                        w_3d_name,
                        r_3d_name,
                        b_2d_name,
                        String::new(),
                        h_3d_name,
                    ],
                    output: vec![gru_y_name, gru_y_h_name.clone()],
                    name: op_name.clone(),
                    op_type: "GRU".to_string(),
                    attribute: gru_attrs,
                    ..Default::default()
                });

                nodes.push(NodeProto {
                    input: vec![gru_y_h_name, axes0_name.clone()],
                    output: vec![final_output_name],
                    name: format!("{}_squeeze_h", op_name),
                    op_type: "Squeeze".to_string(),
                    ..Default::default()
                });
            } else if op.op_type.eq_ignore_ascii_case("tile") {
                // ONNX Tile takes repeats as a second input tensor (INT64)
                let data_input = if let Some(data_id) = op.input_operands.first() {
                    operand_name(graph, *data_id)
                } else {
                    return Err(Self::invalid_operand(
                        "tile missing data input",
                        idx as u32,
                        Some((op, idx)),
                    ));
                };

                // Repeats come from attribute; ignore missing second operand and synthesize.
                let repeats: Vec<i64> =
                    op.attributes
                        .get("repetitions")
                        .or_else(|| op.attributes.get("repeats"))
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
                        .ok_or_else(|| {
                            let operand_id =
                                op.input_operands.get(1).copied().unwrap_or_else(|| {
                                    op.input_operands.first().copied().unwrap_or(0)
                                });
                            Self::invalid_operand(
                                "tile repeats/repetitions attribute",
                                operand_id,
                                Some((op, idx)),
                            )
                        })?;

                // If all repeats are 1, Tile is a no-op. Emit Identity to avoid shape issues.
                if repeats.is_empty() || repeats.iter().all(|&r| r == 1) {
                    nodes.push(NodeProto {
                        input: vec![data_input],
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: format!("{}_identity", op_name),
                        op_type: "Identity".to_string(),
                        attribute: vec![],
                        ..Default::default()
                    });
                    continue;
                }

                let repeats_name = format!("{}_repeats", op_name);
                initializers.push(TensorProto {
                    name: repeats_name.clone(),
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![repeats.len() as i64],
                    int64_data: repeats,
                    ..Default::default()
                });
                let inputs = vec![data_input, repeats_name];

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: vec![],
                    ..Default::default()
                });
            } else if op.op_type == "clamp" {
                // Clamp (Clip in ONNX) uses min/max as inputs (not attributes) in opset 11+.
                // Preserve WebNN defaults (float: -inf/+inf) and ignore NaN bounds.
                if op.input_operands.is_empty() {
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!(
                            "clamp requires at least 1 input, got 0 in operation {op_name}"
                        ),
                    });
                }
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                let input_operand = graph.operand(op.input_operands[0]).ok_or_else(|| {
                    Self::invalid_operand(
                        "clamp input lookup",
                        op.input_operands[0],
                        Some((op, idx)),
                    )
                })?;
                let input_dtype = input_operand.descriptor.data_type;
                let onnx_dtype = Self::data_type_code(input_dtype);

                let min_value_raw = Self::parse_f64_attr(op.attributes.get("minValue"));
                let max_value_raw = Self::parse_f64_attr(op.attributes.get("maxValue"));
                let min_value = min_value_raw.and_then(|v| if v.is_nan() { None } else { Some(v) });
                let max_value = max_value_raw.and_then(|v| if v.is_nan() { None } else { Some(v) });
                let is_float_input = matches!(input_dtype, DataType::Float32 | DataType::Float16);
                let min_value = if is_float_input {
                    Some(min_value.unwrap_or(f64::NEG_INFINITY))
                } else {
                    min_value
                };
                let max_value = if is_float_input {
                    Some(max_value.unwrap_or(f64::INFINITY))
                } else {
                    max_value
                };

                if let Some(min_value) = min_value {
                    let min_name = format!("{}_min", op_name);
                    inputs.push(min_name.clone());
                    let min_tensor = match input_dtype {
                        DataType::Float32 => TensorProto {
                            name: min_name,
                            data_type: ProtoDataType::Float as i32,
                            dims: vec![],
                            raw_data: (min_value as f32).to_le_bytes().to_vec(),
                            ..Default::default()
                        },
                        DataType::Float16 => {
                            let bits = half::f16::from_f32(min_value as f32).to_bits();
                            TensorProto {
                                name: min_name,
                                data_type: ProtoDataType::Float16 as i32,
                                dims: vec![],
                                raw_data: bits.to_le_bytes().to_vec(),
                                ..Default::default()
                            }
                        }
                        DataType::Int64 => TensorProto {
                            name: min_name,
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![],
                            int64_data: vec![min_value as i64],
                            ..Default::default()
                        },
                        DataType::Uint64 => TensorProto {
                            name: min_name,
                            data_type: ProtoDataType::Uint64 as i32,
                            dims: vec![],
                            uint64_data: vec![min_value as u64],
                            ..Default::default()
                        },
                        _ => {
                            Self::create_scalar_initializer(min_name, onnx_dtype, min_value as f32)
                        }
                    };
                    initializers.push(min_tensor);
                }

                if max_value.is_some() && min_value.is_none() {
                    // ONNX Clip requires empty min placeholder when only max is provided.
                    inputs.push(String::new());
                }
                if let Some(max_value) = max_value {
                    let max_name = format!("{}_max", op_name);
                    inputs.push(max_name.clone());
                    let max_tensor = match input_dtype {
                        DataType::Float32 => TensorProto {
                            name: max_name,
                            data_type: ProtoDataType::Float as i32,
                            dims: vec![],
                            raw_data: (max_value as f32).to_le_bytes().to_vec(),
                            ..Default::default()
                        },
                        DataType::Float16 => {
                            let bits = half::f16::from_f32(max_value as f32).to_bits();
                            TensorProto {
                                name: max_name,
                                data_type: ProtoDataType::Float16 as i32,
                                dims: vec![],
                                raw_data: bits.to_le_bytes().to_vec(),
                                ..Default::default()
                            }
                        }
                        DataType::Int64 => TensorProto {
                            name: max_name,
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![],
                            int64_data: vec![max_value as i64],
                            ..Default::default()
                        },
                        DataType::Uint64 => TensorProto {
                            name: max_name,
                            data_type: ProtoDataType::Uint64 as i32,
                            dims: vec![],
                            uint64_data: vec![max_value as u64],
                            ..Default::default()
                        },
                        _ => {
                            Self::create_scalar_initializer(max_name, onnx_dtype, max_value as f32)
                        }
                    };
                    initializers.push(max_tensor);
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: vec![],
                    ..Default::default()
                });
            } else if op.op_type == "reshape" {
                // Reshape requires shape as a second input tensor in ONNX (not as an attribute)
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Handle newShape attribute - can be array (static), string (operand reference), or missing
                if let Some(new_shape_attr) = op.attributes.get("newShape") {
                    if let Some(shape_dims) = Self::parse_dimension_array(new_shape_attr) {
                        // Case 1: newShape is an array (static or dynamic)
                        let has_dynamic = shape_dims
                            .iter()
                            .any(|d| matches!(d, Dimension::Dynamic(_)));
                        if has_dynamic {
                            let runtime_shape_name = Self::build_runtime_shape_input(
                                &format!("{}_shape", op_name),
                                &shape_dims,
                                graph,
                                op,
                                &mut nodes,
                                &mut initializers,
                            );
                            inputs.push(runtime_shape_name);
                        } else {
                            let shape_values: Vec<i64> = shape_dims
                                .iter()
                                .map(|d| get_static_or_max_size(d) as i64)
                                .collect();
                            let shape_name = format!("{}_shape", op_name);
                            inputs.push(shape_name.clone());

                            initializers.push(TensorProto {
                                name: shape_name,
                                data_type: ProtoDataType::Int64 as i32,
                                dims: vec![shape_values.len() as i64],
                                int64_data: shape_values,
                                ..Default::default()
                            });
                        }
                    } else if let Some(shape_operand_name) = new_shape_attr.as_str() {
                        // Case 2: newShape is a string (operand reference) - use referenced operand as second input
                        // This handles dynamic reshapes where the shape is computed at runtime

                        // Use the string as-is since operand names in the graph preserve their original format
                        // The loader's sanitization only affects certain identifier patterns (not output names on LHS of =)
                        inputs.push(shape_operand_name.to_string());
                    } else {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!(
                                "Reshape operation has invalid newShape attribute type (not array or string) in operation {}",
                                op_name
                            ),
                        });
                    }
                } else {
                    // Case 3: No newShape attribute - infer from output operand descriptor
                    let output_id = op.output_operand.expect("Single-output operation expected");
                    let output_operand = graph.operand(output_id).ok_or_else(|| {
                        Self::invalid_operand("reshape output lookup", output_id, Some((op, idx)))
                    })?;
                    let shape_dims = output_operand.descriptor.shape.clone();
                    let has_dynamic = shape_dims
                        .iter()
                        .any(|d| matches!(d, Dimension::Dynamic(_)));
                    if has_dynamic {
                        let runtime_shape_name = Self::build_runtime_shape_input(
                            &format!("{}_shape", op_name),
                            &shape_dims,
                            graph,
                            op,
                            &mut nodes,
                            &mut initializers,
                        );
                        inputs.push(runtime_shape_name);
                    } else {
                        let shape_values: Vec<i64> = shape_dims
                            .iter()
                            .map(|d| get_static_or_max_size(d) as i64)
                            .collect();

                        let shape_name = format!("{}_shape", op_name);
                        inputs.push(shape_name.clone());
                        initializers.push(TensorProto {
                            name: shape_name,
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![shape_values.len() as i64],
                            int64_data: shape_values,
                            ..Default::default()
                        });
                    }
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: vec![], // No attributes for Reshape
                    ..Default::default()
                });
            } else if op.op_type == "expand" {
                // WebNN expand has two variants:
                // 1. With 'axes' - adds dimensions (maps to ONNX Unsqueeze)
                // 2. With 'newShape' - expands shape (maps to ONNX Expand or Reshape)

                debug_print!("[DEBUG] Processing WebNN expand operation:");
                debug_print!("  Op name: {}", op_name);
                if let Some(obj) = op.attributes.as_object() {
                    debug_print!("  Attributes: {:?}", obj.keys().collect::<Vec<_>>());
                }

                if let Some(axes_val) = op.attributes.get("axes").and_then(|v| v.as_array()) {
                    // WebNN expand with axes -> ONNX Unsqueeze
                    // In ONNX opset 13+, axes must be provided as an input tensor, not attribute
                    let axes_values: Vec<i64> =
                        axes_val.iter().filter_map(|v| v.as_i64()).collect();

                    let mut inputs: Vec<String> = op
                        .input_operands
                        .iter()
                        .map(|id| operand_name(graph, *id))
                        .collect();

                    // Create axes tensor name and add as second input
                    let axes_name = format!("{}_axes", op_name);
                    inputs.push(axes_name.clone());

                    // Add axes as an initializer (constant tensor)
                    initializers.push(TensorProto {
                        name: axes_name,
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![axes_values.len() as i64], // 1D tensor
                        int64_data: axes_values,
                        ..Default::default()
                    });

                    nodes.push(NodeProto {
                        input: inputs,
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: op_name.clone(),
                        op_type: "Unsqueeze".to_string(),
                        attribute: vec![], // No attributes for Unsqueeze in opset 13+
                        ..Default::default()
                    });
                } else if let Some(new_shape) =
                    op.attributes.get("newShape").and_then(|v| v.as_array())
                {
                    // WebNN expand with newShape can be either:
                    // 1. ONNX Expand (broadcasting-compatible shapes)
                    // 2. ONNX Reshape (arbitrary shape changes)

                    let mut inputs: Vec<String> = op
                        .input_operands
                        .iter()
                        .map(|id| operand_name(graph, *id))
                        .collect();

                    let target_dims =
                        Self::parse_dimension_array(&serde_json::Value::Array(new_shape.clone()))
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "onnx".to_string(),
                                reason: format!(
                                    "Expand newShape contains invalid or unsupported dimensions in operation {}",
                                    op_name
                                ),
                            })?;
                    let shape_values: Vec<i64> = target_dims
                        .iter()
                        .map(|d| get_static_or_max_size(d) as i64)
                        .collect();
                    let has_dynamic_shape = target_dims
                        .iter()
                        .any(|d| matches!(d, Dimension::Dynamic(_)));

                    // Get input operand shape to determine if this is broadcasting or reshaping
                    let input_id = op.input_operands[0];
                    // Use tracked shape if available, otherwise fall back to descriptor
                    let input_shape = operand_shapes.get(&input_id).cloned().unwrap_or_else(|| {
                        graph.operands[input_id as usize]
                            .descriptor
                            .static_or_max_shape()
                    });

                    // Check if shapes are broadcasting-compatible (ONNX Expand rules):
                    // - Align shapes from the right
                    // - Each dimension pair must be equal or one must be 1
                    // - SPECIAL CASE: Scalars (rank 0) from constants should use Reshape, not Expand
                    let is_broadcast_compatible = {
                        let mut compatible = true;
                        let input_rank = input_shape.len();
                        let target_rank = shape_values.len();

                        debug_print!("[DEBUG] Expand operation:");
                        debug_print!("  Op name: {}", op_name);
                        debug_print!("  Input operand ID: {}", input_id);
                        debug_print!("  Input shape: {:?} (rank={})", input_shape, input_rank);
                        debug_print!("  Target shape: {:?} (rank={})", shape_values, target_rank);

                        // Only apply scalar handling to actual constant operands
                        // Runtime computed values may have different shapes than static descriptors
                        let is_constant = graph
                            .constant_operand_ids_to_handles
                            .contains_key(&input_id);

                        // Scalars (rank 0) need special handling regardless of whether they're constants:
                        // Reshape to target rank with all 1s, then expand will broadcast properly
                        if input_rank == 0 {
                            debug_print!(
                                "  Scalar input (constant={}) - will reshape to match target rank with all 1s",
                                is_constant
                            );

                            // Step 1: Reshape scalar to [1,1,...,1] with same rank as target
                            let reshape_intermediate =
                                format!("{}_scalar_to_rank{}", op_name, target_rank);
                            let reshape_shape_name = format!("{}_reshape_shape", op_name);

                            // Create shape [1,1,...,1] with target_rank dimensions
                            let intermediate_shape = vec![1i64; target_rank];

                            initializers.push(TensorProto {
                                name: reshape_shape_name.clone(),
                                data_type: ProtoDataType::Int64 as i32,
                                dims: vec![target_rank as i64],
                                int64_data: intermediate_shape,
                                ..Default::default()
                            });

                            nodes.push(NodeProto {
                                input: vec![inputs[0].clone(), reshape_shape_name],
                                output: vec![reshape_intermediate.clone()],
                                name: format!("{}_scalar_reshape", op_name),
                                op_type: "Reshape".to_string(),
                                attribute: vec![],
                                ..Default::default()
                            });

                            // Step 2: Update input for subsequent Expand to use reshaped tensor
                            inputs[0] = reshape_intermediate;

                            // Now it's compatible for expand (from [1,1,...,1] to target shape)
                            compatible = true;
                        } else {
                            // Align from right and check each dimension
                            for i in 0..input_rank.min(target_rank) {
                                let input_dim = input_shape[input_rank - 1 - i];
                                let target_dim = shape_values[target_rank - 1 - i] as u32;

                                // Dimensions are compatible if they're equal or either is 1
                                if input_dim != target_dim && input_dim != 1 && target_dim != 1 {
                                    debug_print!(
                                        "  Incompatible at dim {}: {} vs {}",
                                        i,
                                        input_dim,
                                        target_dim
                                    );
                                    compatible = false;
                                    break;
                                }
                            }
                        }

                        debug_print!("  Broadcasting compatible: {}", compatible);
                        compatible
                    };

                    // For dynamic target shapes, keep ONNX Expand semantics.
                    // Using Reshape here can break runtime broadcasting when real dims
                    // differ from max-size placeholders.
                    let op_type = if has_dynamic_shape || is_broadcast_compatible {
                        "Expand"
                    } else {
                        debug_print!(
                            "[FIX] Using Reshape instead of Expand for {} -> {:?}",
                            op_name,
                            shape_values
                        );
                        "Reshape"
                    };

                    let shape_name = format!("{}_shape", op_name);
                    if has_dynamic_shape {
                        let runtime_shape_name = Self::build_runtime_shape_input(
                            &shape_name,
                            &target_dims,
                            graph,
                            op,
                            &mut nodes,
                            &mut initializers,
                        );
                        inputs.push(runtime_shape_name);
                    } else {
                        inputs.push(shape_name.clone());
                    }

                    // Update operand_shapes with the output shape before moving shape_values
                    if let Some(output_id) = op.output_operand {
                        let output_shape: Vec<u32> =
                            shape_values.iter().map(|&v| v as u32).collect();
                        operand_shapes.insert(output_id, output_shape);
                    }

                    if !has_dynamic_shape {
                        // Add shape as an initializer (constant tensor)
                        initializers.push(TensorProto {
                            name: shape_name,
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![shape_values.len() as i64], // 1D tensor
                            int64_data: shape_values,
                            ..Default::default()
                        });
                    }

                    nodes.push(NodeProto {
                        input: inputs,
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: op_name,
                        op_type: op_type.to_string(),
                        attribute: vec![], // No attributes for Expand or Reshape
                        ..Default::default()
                    });
                } else {
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!(
                            "Expand operation requires either 'axes' or 'newShape' attribute in operation {}",
                            op_name
                        ),
                    });
                }
            } else if op.op_type.starts_with("reduce") {
                // In ONNX opset 13, only ReduceSum supports axes as input tensor.
                let supports_axes_as_input = matches!(op.op_type.as_str(), "reduceSum");

                // Check if input needs casting (uint32 not supported by ONNX Runtime for some reductions)
                let input_id = op.input_operands[0];
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("reduction input lookup", input_id, Some((op, idx)))
                })?;

                let input_name = operand_name(graph, input_id);
                let final_output_name = operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );
                let axes_i64 = Self::parse_i64_array(op, "axes");

                // Match existing WPT behavior for empty axes handling.
                if axes_i64.as_ref().is_some_and(|axes| axes.is_empty()) {
                    let (special_op, special_inputs): (&str, Vec<String>) = match op
                        .op_type
                        .as_str()
                    {
                        "reduceL1" | "reduceL2" => ("Abs", vec![input_name.clone()]),
                        "reduceSumSquare" => ("Mul", vec![input_name.clone(), input_name.clone()]),
                        "reduceLogSum" => ("Log", vec![input_name.clone()]),
                        _ => ("Identity", vec![input_name.clone()]),
                    };
                    nodes.push(NodeProto {
                        input: special_inputs,
                        output: vec![final_output_name],
                        name: op_name.clone(),
                        op_type: special_op.to_string(),
                        ..Default::default()
                    });
                    continue;
                }

                let needs_cast = matches!(
                    input_operand.descriptor.data_type,
                    DataType::Uint32 | DataType::Uint8
                );

                let actual_input_name = if needs_cast {
                    // Cast uint32/uint8 to float32 for reduction operations
                    let cast_output = format!("{}_cast_to_float", op_name);
                    nodes.push(NodeProto {
                        input: vec![input_name],
                        output: vec![cast_output.clone()],
                        name: format!("{}_pre_cast", op_name),
                        op_type: "Cast".to_string(),
                        attribute: vec![AttributeProto {
                            name: "to".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: ProtoDataType::Float as i32 as i64,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    cast_output
                } else {
                    input_name
                };

                let mut inputs: Vec<String> = vec![actual_input_name];
                // Add any additional inputs (though reductions typically have only one input)
                inputs.extend(
                    op.input_operands[1..]
                        .iter()
                        .map(|id| operand_name(graph, *id)),
                );

                let mut attributes = Vec::new();

                // Extract axes from attributes.
                if let Some(axes_i64) = axes_i64 {
                    if supports_axes_as_input {
                        let axes_name = format!("{}_axes", op_name);
                        inputs.push(axes_name.clone());
                        initializers.push(TensorProto {
                            name: axes_name,
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![axes_i64.len() as i64],
                            int64_data: axes_i64,
                            ..Default::default()
                        });
                    } else {
                        attributes.push(AttributeProto {
                            name: "axes".to_string(),
                            r#type: AttributeType::Ints as i32,
                            ints: axes_i64,
                            ..Default::default()
                        });
                    }
                }

                // WebNN default is keepDimensions=false; ONNX default is keepdims=1.
                // Always set it explicitly to avoid scalar-shape mismatches.
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                attributes.push(AttributeProto {
                    name: "keepdims".to_string(),
                    r#type: AttributeType::Int as i32,
                    i: if keep_dims { 1 } else { 0 },
                    ..Default::default()
                });

                let reduce_output_name = if needs_cast {
                    // Output to temporary name, will cast back after
                    format!("{}_float_output", op_name)
                } else {
                    final_output_name.clone()
                };

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![reduce_output_name.clone()],
                    name: op_name.clone(),
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: attributes,
                    ..Default::default()
                });

                // Cast back to original type if needed
                if needs_cast {
                    let original_type = Self::data_type_code(input_operand.descriptor.data_type);
                    nodes.push(NodeProto {
                        input: vec![reduce_output_name],
                        output: vec![final_output_name],
                        name: format!("{}_post_cast", op_name),
                        op_type: "Cast".to_string(),
                        attribute: vec![AttributeProto {
                            name: "to".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: original_type as i32 as i64,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                }
            } else if op.op_type.eq_ignore_ascii_case("pad") {
                let data_input_id = op.input_operands[0];
                let data_input_name = operand_name(graph, data_input_id);
                let output_name = operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );

                let input_rank = graph
                    .operand(data_input_id)
                    .map(|operand| operand.descriptor.shape.len())
                    .unwrap_or(0);
                if input_rank == 0 {
                    // ORT Pad rejects scalars; WebNN semantics for empty paddings is no-op.
                    nodes.push(NodeProto {
                        input: vec![data_input_name],
                        output: vec![output_name],
                        name: format!("{}_scalar_identity", op_name),
                        op_type: "Identity".to_string(),
                        attribute: vec![],
                        ..Default::default()
                    });
                    continue;
                }
                let pads_data: Vec<i64> = if let (Some(begin), Some(end)) = (
                    Self::parse_i64_array_any(op, &["beginningPadding", "beginning_padding"]),
                    Self::parse_i64_array_any(op, &["endingPadding", "ending_padding"]),
                ) {
                    let mut combined = begin;
                    combined.extend(end);
                    combined
                } else if let Some(p) =
                    Self::parse_i64_array_any(op, &["padding", "pads", "paddings"])
                {
                    p
                } else {
                    vec![0; input_rank.saturating_mul(2)]
                };

                let pads_name = format!("{}_pads", op_name);
                initializers.push(TensorProto {
                    name: pads_name.clone(),
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![pads_data.len() as i64],
                    int64_data: pads_data,
                    ..Default::default()
                });

                let mut inputs = vec![data_input_name, pads_name];
                let mode = op
                    .attributes
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("constant")
                    .to_ascii_lowercase();
                let onnx_mode = match mode.as_str() {
                    "edge" => "edge",
                    "reflection" => "reflect",
                    _ => "constant",
                };
                if onnx_mode == "constant" {
                    let data_dtype = graph
                        .operand(data_input_id)
                        .map(|operand| Self::data_type_code(operand.descriptor.data_type))
                        .unwrap_or(ProtoDataType::Float);
                    let value = Self::parse_f64_attr(op.attributes.get("value")).unwrap_or(0.0);
                    let value_name = format!("{}_pad_value", op_name);
                    let value_tensor = match data_dtype {
                        ProtoDataType::Float => TensorProto {
                            name: value_name.clone(),
                            data_type: ProtoDataType::Float as i32,
                            dims: vec![],
                            raw_data: (value as f32).to_le_bytes().to_vec(),
                            ..Default::default()
                        },
                        ProtoDataType::Float16 => {
                            let bits = half::f16::from_f32(value as f32).to_bits();
                            TensorProto {
                                name: value_name.clone(),
                                data_type: ProtoDataType::Float16 as i32,
                                dims: vec![],
                                raw_data: bits.to_le_bytes().to_vec(),
                                ..Default::default()
                            }
                        }
                        ProtoDataType::Int64 => TensorProto {
                            name: value_name.clone(),
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![],
                            int64_data: vec![value as i64],
                            ..Default::default()
                        },
                        ProtoDataType::Uint64 => TensorProto {
                            name: value_name.clone(),
                            data_type: ProtoDataType::Uint64 as i32,
                            dims: vec![],
                            uint64_data: vec![value as u64],
                            ..Default::default()
                        },
                        _ => Self::create_scalar_initializer(
                            value_name.clone(),
                            data_dtype,
                            value as f32,
                        ),
                    };
                    initializers.push(value_tensor);
                    inputs.push(value_name);
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![output_name],
                    name: op_name,
                    op_type: "Pad".to_string(),
                    attribute: vec![AttributeProto {
                        name: "mode".to_string(),
                        r#type: AttributeType::String as i32,
                        s: onnx_mode.as_bytes().to_vec(),
                        ..Default::default()
                    }],
                    ..Default::default()
                });
            } else if op.op_type.eq_ignore_ascii_case("resample2d") {
                let input_id = op.input_operands[0];
                let input_name = operand_name(graph, input_id);
                let output_id = op
                    .output_operand
                    .ok_or(GraphError::InvalidConversionOperand { operand: 0 })?;
                let output_name = operand_name(graph, output_id);
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("resample2d input lookup", input_id, Some((op, idx)))
                })?;
                let input_shape = operand_shapes
                    .get(&input_id)
                    .cloned()
                    .unwrap_or_else(|| input_operand.descriptor.static_or_max_shape());
                let rank = input_shape.len();
                if rank == 0 {
                    nodes.push(NodeProto {
                        input: vec![input_name],
                        output: vec![output_name],
                        name: op_name,
                        op_type: "Identity".to_string(),
                        ..Default::default()
                    });
                    continue;
                }

                let axes: Vec<usize> = Self::parse_i64_array(op, "axes")
                    .unwrap_or_else(|| vec![rank as i64 - 2, rank as i64 - 1])
                    .into_iter()
                    .map(|a| {
                        if a < 0 {
                            (rank as i64 + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                if axes.len() != 2 || axes.iter().any(|&a| a >= rank) {
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!(
                            "resample2d axes must contain 2 valid axes for rank {}: {:?}",
                            rank, axes
                        ),
                    });
                }

                let sizes_attr: Option<Vec<i64>> = op.attributes.get("sizes").and_then(|v| {
                    v.as_array().map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_i64().or_else(|| x.as_u64().map(|u| u as i64)))
                            .collect::<Vec<_>>()
                    })
                });
                let scales_attr: Option<Vec<f64>> = op.attributes.get("scales").and_then(|v| {
                    v.as_array().map(|arr| {
                        arr.iter()
                            .filter_map(|x| Self::parse_f64_attr(Some(x)))
                            .collect::<Vec<_>>()
                    })
                });

                // sizes takes precedence over scales, matching WebNN expectation.
                let spatial_sizes = if let Some(sizes) = sizes_attr {
                    if sizes.len() != 2 {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!("resample2d sizes must have length 2, got {:?}", sizes),
                        });
                    }
                    sizes
                } else if let Some(scales) = scales_attr {
                    if scales.len() != 2 {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!(
                                "resample2d scales must have length 2, got {:?}",
                                scales
                            ),
                        });
                    }
                    let mut sizes = Vec::with_capacity(2);
                    for (axis_idx, scale) in scales.iter().enumerate() {
                        let in_dim = input_shape[axes[axis_idx]].max(1) as f64;
                        let out_dim = (in_dim * *scale).round().max(1.0) as i64;
                        sizes.push(out_dim);
                    }
                    sizes
                } else {
                    vec![input_shape[axes[0]] as i64, input_shape[axes[1]] as i64]
                };

                let mut output_sizes: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
                output_sizes[axes[0]] = spatial_sizes[0];
                output_sizes[axes[1]] = spatial_sizes[1];

                // Provide only scales input to avoid ORT ambiguity when both scales/sizes exist.
                let scales: Vec<f32> = output_sizes
                    .iter()
                    .enumerate()
                    .map(|(i, &out_dim)| {
                        let in_dim = input_shape[i].max(1) as f32;
                        out_dim as f32 / in_dim
                    })
                    .collect();
                let scales_name = format!("{}_scales", op_name);
                initializers.push(TensorProto {
                    name: scales_name.clone(),
                    data_type: ProtoDataType::Float as i32,
                    dims: vec![scales.len() as i64],
                    float_data: scales,
                    ..Default::default()
                });

                let mode = op
                    .attributes
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("nearest-neighbor")
                    .to_ascii_lowercase();
                let onnx_mode = if mode == "linear" {
                    "linear"
                } else {
                    "nearest"
                };
                let coord_mode = if onnx_mode == "linear" {
                    "half_pixel"
                } else {
                    "asymmetric"
                };

                nodes.push(NodeProto {
                    input: vec![input_name, String::new(), scales_name],
                    output: vec![output_name],
                    name: op_name,
                    op_type: "Resize".to_string(),
                    attribute: vec![
                        AttributeProto {
                            name: "mode".to_string(),
                            r#type: AttributeType::String as i32,
                            s: onnx_mode.as_bytes().to_vec(),
                            ..Default::default()
                        },
                        AttributeProto {
                            name: "coordinate_transformation_mode".to_string(),
                            r#type: AttributeType::String as i32,
                            s: coord_mode.as_bytes().to_vec(),
                            ..Default::default()
                        },
                        AttributeProto {
                            name: "nearest_mode".to_string(),
                            r#type: AttributeType::String as i32,
                            s: b"floor".to_vec(),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                });
            } else if op.op_type.eq_ignore_ascii_case("gathernd")
                || op.op_type.eq_ignore_ascii_case("gatherelements")
            {
                if op.op_type.eq_ignore_ascii_case("gatherelements") {
                    // GatherElements: cast indices to int64 and clamp to valid range.
                    let data_id = op.input_operands[0];
                    let indices_id = op.input_operands[1];
                    let data_name = operand_name(graph, data_id);
                    let indices_name = operand_name(graph, indices_id);

                    let data_operand = graph.operand(data_id).ok_or_else(|| {
                        Self::invalid_operand(
                            "gatherElements data lookup",
                            data_id,
                            Some((op, idx)),
                        )
                    })?;

                    let rank = data_operand.descriptor.shape.len();
                    if rank == 0 {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: "gatherElements requires rank >= 1 input".to_string(),
                        });
                    }
                    let mut axis = op
                        .attributes
                        .get("axis")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    if axis < 0 {
                        axis += rank as i64;
                    }
                    if axis < 0 || axis as usize >= rank {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!(
                                "gatherElements axis {} out of bounds for rank {}",
                                axis, rank
                            ),
                        });
                    }
                    let dim_size =
                        get_static_or_max_size(&data_operand.descriptor.shape[axis as usize])
                            as i64;

                    let indices_i64 = format!("{}_indices_i64", op_name);
                    nodes.push(Self::create_cast_node(
                        &format!("{}_indices_cast", op_name),
                        indices_name,
                        indices_i64.clone(),
                        ProtoDataType::Int64,
                    ));

                    let min_name = format!("{}_indices_min", op_name);
                    let max_name = format!("{}_indices_max", op_name);
                    let clamped_name = format!("{}_indices_clamped", op_name);
                    initializers.push(TensorProto {
                        name: min_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![],
                        int64_data: vec![-dim_size],
                        ..Default::default()
                    });
                    initializers.push(TensorProto {
                        name: max_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![],
                        int64_data: vec![dim_size - 1],
                        ..Default::default()
                    });
                    nodes.push(NodeProto {
                        input: vec![indices_i64, min_name, max_name],
                        output: vec![clamped_name.clone()],
                        name: format!("{}_indices_clip", op_name),
                        op_type: "Clip".to_string(),
                        ..Default::default()
                    });

                    nodes.push(NodeProto {
                        input: vec![data_name, clamped_name],
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: op_name,
                        op_type: "GatherElements".to_string(),
                        attribute: vec![AttributeProto {
                            name: "axis".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: axis,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                } else {
                    // GatherND: cast indices to int64 and clamp each component.
                    let data_id = op.input_operands[0];
                    let indices_id = op.input_operands[1];
                    let data_name = operand_name(graph, data_id);
                    let indices_name = operand_name(graph, indices_id);

                    let data_operand = graph.operand(data_id).ok_or_else(|| {
                        Self::invalid_operand("gatherND data lookup", data_id, Some((op, idx)))
                    })?;
                    let indices_operand = graph.operand(indices_id).ok_or_else(|| {
                        Self::invalid_operand(
                            "gatherND indices lookup",
                            indices_id,
                            Some((op, idx)),
                        )
                    })?;

                    let k = indices_operand
                        .descriptor
                        .shape
                        .last()
                        .map(get_static_or_max_size)
                        .unwrap_or(1) as usize;
                    let data_rank = data_operand.descriptor.shape.len();
                    if k == 0 || k > data_rank {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!(
                                "gatherND invalid indices last dim {} for data rank {}",
                                k, data_rank
                            ),
                        });
                    }

                    let indices_i64 = format!("{}_indices_i64", op_name);
                    nodes.push(Self::create_cast_node(
                        &format!("{}_indices_cast", op_name),
                        indices_name,
                        indices_i64.clone(),
                        ProtoDataType::Int64,
                    ));

                    let mins: Vec<i64> = data_operand.descriptor.shape[..k]
                        .iter()
                        .map(|d| -(get_static_or_max_size(d) as i64))
                        .collect();
                    let maxs: Vec<i64> = data_operand.descriptor.shape[..k]
                        .iter()
                        .map(|d| get_static_or_max_size(d) as i64 - 1)
                        .collect();

                    let min_name = format!("{}_indices_min", op_name);
                    let max_name = format!("{}_indices_max", op_name);
                    let clamped_low_name = format!("{}_indices_clamped_low", op_name);
                    let clamped_name = format!("{}_indices_clamped", op_name);
                    initializers.push(TensorProto {
                        name: min_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![k as i64],
                        int64_data: mins,
                        ..Default::default()
                    });
                    initializers.push(TensorProto {
                        name: max_name.clone(),
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![k as i64],
                        int64_data: maxs,
                        ..Default::default()
                    });
                    nodes.push(NodeProto {
                        input: vec![indices_i64, min_name],
                        output: vec![clamped_low_name.clone()],
                        name: format!("{}_indices_max", op_name),
                        op_type: "Max".to_string(),
                        ..Default::default()
                    });
                    nodes.push(NodeProto {
                        input: vec![clamped_low_name, max_name],
                        output: vec![clamped_name.clone()],
                        name: format!("{}_indices_min", op_name),
                        op_type: "Min".to_string(),
                        ..Default::default()
                    });

                    let mut attrs = Vec::new();
                    if let Some(batch_dims) = op
                        .attributes
                        .get("batchDimensions")
                        .and_then(|v| v.as_i64())
                    {
                        attrs.push(AttributeProto {
                            name: "batch_dims".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: batch_dims,
                            ..Default::default()
                        });
                    }

                    nodes.push(NodeProto {
                        input: vec![data_name, clamped_name],
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: op_name,
                        op_type: "GatherND".to_string(),
                        attribute: attrs,
                        ..Default::default()
                    });
                }
            } else if op.op_type == "slice" {
                // Slice operation - ONNX requires starts, ends, axes, steps as input tensors
                // Special case: ONNX Runtime doesn't support slicing 0D tensors

                // Check if input is 0D (scalar)
                let input_operand_id = op.input_operands[0];
                let input_operand = graph.operand(input_operand_id).ok_or_else(|| {
                    Self::invalid_operand("slice input lookup", input_operand_id, Some((op, idx)))
                })?;
                let is_0d = input_operand.descriptor.shape.is_empty();

                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Extract starts, sizes/ends, axes, and steps from attributes
                let parse_attr_i64 = |key: &str| -> Option<Vec<i64>> {
                    op.attributes.get(key).and_then(|v| {
                        if let Some(arr) = v.as_array() {
                            let vals: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
                            if !vals.is_empty() { Some(vals) } else { None }
                        } else if let Some(name) = v.as_str() {
                            get_const_i64(name)
                        } else {
                            None
                        }
                    })
                };

                let starts = parse_attr_i64("starts").unwrap_or_default();
                // Prefer explicit ends attribute; fall back to sizes (ONNX/WebNN use sizes)
                let mut ends = parse_attr_i64("ends").unwrap_or_default();
                let sizes = parse_attr_i64("sizes").unwrap_or_default();

                // Special case: 0D tensor (scalar) cannot be sliced
                // Use Identity node instead (ONNX Runtime doesn't support slicing scalars)
                // A scalar has no axes, so any slice operation should just return the scalar unchanged
                if is_0d {
                    nodes.push(NodeProto {
                        input: vec![inputs[0].clone()],
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: op_name,
                        op_type: "Identity".to_string(),
                        ..Default::default()
                    });
                    continue; // Skip the rest of the slice handling
                }

                let axes = parse_attr_i64("axes");

                let steps = parse_attr_i64("strides");

                // Convert sizes to ends when explicit ends are not provided: ends[i] = starts[i] + sizes[i]
                if ends.is_empty() {
                    ends = starts
                        .iter()
                        .zip(sizes.iter())
                        .map(|(start, size)| start + size)
                        .collect();
                }

                // Capture length before moving starts
                let starts_len = starts.len();

                // Add starts as initializer
                let starts_name = format!("{}_starts", op_name);
                inputs.push(starts_name.clone());
                initializers.push(TensorProto {
                    name: starts_name,
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![starts_len as i64],
                    int64_data: starts,
                    ..Default::default()
                });

                // Add ends as initializer
                let ends_name = format!("{}_ends", op_name);
                inputs.push(ends_name.clone());
                initializers.push(TensorProto {
                    name: ends_name,
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![ends.len() as i64],
                    int64_data: ends,
                    ..Default::default()
                });

                // Add axes as initializer
                // If axes not provided, default to [0, 1, ..., len(starts)-1]
                let axes_data = axes.unwrap_or_else(|| (0..starts_len as i64).collect());

                let axes_name = format!("{}_axes", op_name);
                inputs.push(axes_name.clone());
                initializers.push(TensorProto {
                    name: axes_name,
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![axes_data.len() as i64],
                    int64_data: axes_data,
                    ..Default::default()
                });

                // Add steps as initializer (if provided)
                if let Some(steps_data) = steps {
                    let steps_name = format!("{}_steps", op_name);
                    inputs.push(steps_name.clone());
                    initializers.push(TensorProto {
                        name: steps_name,
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![steps_data.len() as i64],
                        int64_data: steps_data,
                        ..Default::default()
                    });
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: vec![], // No attributes for Slice in opset 13+
                    ..Default::default()
                });
            } else if op.op_type == "split" {
                // Split operation - multi-output operation
                let outputs: Vec<String> = op
                    .output_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Get axis attribute
                let mut attributes = Vec::new();
                if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_u64()) {
                    attributes.push(AttributeProto {
                        name: "axis".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: axis as i64,
                        ..Default::default()
                    });
                }

                // Collect inputs
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Handle splits parameter - either count or sizes
                // ONNX Split opset 13+ takes split sizes as an optional input tensor
                if let Some(splits_val) = op.attributes.get("splits") {
                    if let Some(_count) = splits_val.as_u64() {
                        // Equal splits - ONNX Split without split input divides evenly
                        // based on the number of outputs. No input needed.
                    } else if let Some(sizes) = splits_val.as_array() {
                        // Explicit split sizes - create initializer
                        let split_sizes: Vec<i64> = sizes
                            .iter()
                            .filter_map(|v| v.as_u64().map(|n| n as i64))
                            .collect();

                        let splits_name = format!("{}_splits", op_name);

                        // Create initializer for split sizes
                        let splits_tensor = TensorProto {
                            name: splits_name.clone(),
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![split_sizes.len() as i64],
                            int64_data: split_sizes,
                            ..Default::default()
                        };
                        initializers.push(splits_tensor);

                        // Add splits as second input
                        inputs.push(splits_name);
                    }
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: outputs,
                    name: op_name,
                    op_type: "Split".to_string(),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if op.op_type == "gather" {
                // Gather operation - ONNX only supports int32/int64 indices, need to cast uint32/uint8
                // Also need to clamp indices to prevent out-of-bounds errors (following Chromium's approach)
                let mut inputs: Vec<String> = Vec::new();

                // First input: data tensor
                let data_operand_id = op.input_operands[0];
                inputs.push(operand_name(graph, data_operand_id));

                // Get axis parameter (default is 0)
                let axis = op
                    .attributes
                    .get("axis")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as usize;

                // Get input shape and dimension size at axis
                let data_operand = graph.operand(data_operand_id).ok_or_else(|| {
                    Self::invalid_operand("gather data lookup", data_operand_id, Some((op, idx)))
                })?;

                let data_shape = data_operand.descriptor.static_or_max_shape();

                // Check if axis is within bounds
                if axis >= data_shape.len() {
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!(
                            "Gather operation: axis {} is out of bounds for shape with {} dimensions (operand {})",
                            axis,
                            data_shape.len(),
                            data_operand_id
                        ),
                    });
                }

                let dim_size = data_shape[axis] as i64;

                // Second input: indices tensor - may need casting and clamping
                let indices_id = op.input_operands[1];
                let indices_name = operand_name(graph, indices_id);
                let indices_operand = graph.operand(indices_id).ok_or_else(|| {
                    Self::invalid_operand("gather indices lookup", indices_id, Some((op, idx)))
                })?;

                // Step 1: Cast indices to int64 if needed (required for Clamp operation)
                let indices_after_cast =
                    if !matches!(indices_operand.descriptor.data_type, DataType::Int64) {
                        // Cast to int64 for ONNX compatibility (Clamp requires all inputs to be same type)
                        let cast_output_name = format!("{}_indices_int64", op_name);
                        nodes.push(Self::create_cast_node(
                            &format!("{}_cast_indices", op_name),
                            indices_name,
                            cast_output_name.clone(),
                            ProtoDataType::Int64,
                        ));
                        cast_output_name
                    } else {
                        indices_name
                    };

                // Step 2: Clamp indices to valid range [-dim_size, dim_size - 1]
                // This prevents out-of-bounds errors from ONNX Runtime
                let clamp_min_name = format!("{}_clamp_min", op_name);
                let clamp_max_name = format!("{}_clamp_max", op_name);
                let clamped_indices_name = format!("{}_indices_clamped", op_name);

                // Create scalar initializers for min and max
                initializers.push(TensorProto {
                    name: clamp_min_name.clone(),
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![],
                    int64_data: vec![-dim_size],
                    ..Default::default()
                });

                initializers.push(TensorProto {
                    name: clamp_max_name.clone(),
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![],
                    int64_data: vec![dim_size - 1],
                    ..Default::default()
                });

                // Insert Clip node (Clamp was deprecated in favor of Clip in opset 11+)
                nodes.push(NodeProto {
                    input: vec![indices_after_cast, clamp_min_name, clamp_max_name],
                    output: vec![clamped_indices_name.clone()],
                    name: format!("{}_clip_indices", op_name),
                    op_type: "Clip".to_string(),
                    ..Default::default()
                });

                // ONNX Gather handles indices shape correctly, no reshape needed
                // The output shape is automatically: data.shape[0:axis] + indices.shape + data.shape[axis+1:]
                let final_indices = clamped_indices_name;

                inputs.push(final_indices);

                // Create Gather node
                let attributes = Self::create_operation_attributes(op);
                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: "Gather".to_string(),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if op.op_type == "conv2d" || op.op_type == "convTranspose2d" {
                // Conv2d/ConvTranspose2d operations - handle layout transformations
                let mut conv_inputs: Vec<String> = Vec::new();

                // Handle input layout (NHWC → NCHW if needed)
                let input_name = operand_name(graph, op.input_operands[0]);
                let input_layout = op
                    .attributes
                    .get("inputLayout")
                    .and_then(|v| v.as_str())
                    .unwrap_or("nchw");

                let transposed_input = if input_layout == "nhwc" {
                    // Insert Transpose node: NHWC → NCHW
                    let transpose_output = format!("{}_input_transposed", op_name);
                    nodes.push(NodeProto {
                        input: vec![input_name],
                        output: vec![transpose_output.clone()],
                        name: format!("{}_transpose_input", op_name),
                        op_type: "Transpose".to_string(),
                        attribute: vec![AttributeProto {
                            name: "perm".to_string(),
                            r#type: AttributeType::Ints as i32,
                            ints: vec![0, 3, 1, 2], // NHWC → NCHW
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    transpose_output
                } else {
                    input_name
                };
                conv_inputs.push(transposed_input);

                // Handle filter layout transformation
                let filter_name = operand_name(graph, op.input_operands[1]);
                let filter_layout = op
                    .attributes
                    .get("filterLayout")
                    .and_then(|v| v.as_str())
                    .unwrap_or(if op.op_type == "convTranspose2d" {
                        "iohw"
                    } else {
                        "oihw"
                    });

                let is_transpose = op.op_type == "convTranspose2d";
                let needs_transpose = if is_transpose {
                    // ConvTranspose: ONNX expects IOHW (Input, Output, H, W)
                    filter_layout != "iohw"
                } else {
                    // Conv: ONNX expects OIHW (Output, Input, H, W)
                    filter_layout != "oihw"
                };

                let transposed_filter = if needs_transpose {
                    let perm = if is_transpose {
                        // ConvTranspose filter layout conversions → IOHW
                        match filter_layout {
                            "hwoi" => vec![3, 2, 0, 1], // HWOI (H,W,O,I) → IOHW (I,O,H,W)
                            "ohwi" => vec![3, 0, 1, 2], // OHWI (O,H,W,I) → IOHW (I,O,H,W)
                            "oihw" => vec![1, 0, 2, 3], // OIHW (O,I,H,W) → IOHW (I,O,H,W)
                            _ => vec![0, 1, 2, 3],      // Default: no transpose
                        }
                    } else {
                        // Conv2d filter layout conversions → OIHW
                        match filter_layout {
                            "hwio" => vec![3, 2, 0, 1], // HWIO (H,W,I,O) → OIHW (O,I,H,W)
                            "ohwi" => vec![0, 3, 1, 2], // OHWI (O,H,W,I) → OIHW (O,I,H,W)
                            "ihwo" => vec![3, 0, 1, 2], // IHWO (I,H,W,O) → OIHW (O,I,H,W)
                            _ => vec![0, 1, 2, 3],      // Default: no transpose
                        }
                    };

                    let transpose_output = format!("{}_filter_transposed", op_name);
                    nodes.push(NodeProto {
                        input: vec![filter_name],
                        output: vec![transpose_output.clone()],
                        name: format!("{}_transpose_filter", op_name),
                        op_type: "Transpose".to_string(),
                        attribute: vec![AttributeProto {
                            name: "perm".to_string(),
                            r#type: AttributeType::Ints as i32,
                            ints: perm,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    transpose_output
                } else {
                    filter_name
                };
                conv_inputs.push(transposed_filter);

                // Add bias if present (third input)
                if op.input_operands.len() > 2 {
                    conv_inputs.push(operand_name(graph, op.input_operands[2]));
                }

                // If WebNN layout is NHWC, ONNX node output (NCHW) must be transposed back.
                let final_output_name = operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );
                let conv_output_name = if input_layout.eq_ignore_ascii_case("nhwc") {
                    format!("{}_conv_output_nchw", op_name)
                } else {
                    final_output_name.clone()
                };

                let attributes = Self::create_operation_attributes(op);
                nodes.push(NodeProto {
                    input: conv_inputs,
                    output: vec![conv_output_name.clone()],
                    name: op_name.clone(),
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: attributes,
                    ..Default::default()
                });

                if input_layout.eq_ignore_ascii_case("nhwc") {
                    nodes.push(NodeProto {
                        input: vec![conv_output_name],
                        output: vec![final_output_name],
                        name: format!("{}_transpose_output", op_name),
                        op_type: "Transpose".to_string(),
                        attribute: vec![AttributeProto {
                            name: "perm".to_string(),
                            r#type: AttributeType::Ints as i32,
                            ints: vec![0, 2, 3, 1], // NCHW -> NHWC
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                }
            } else if matches!(
                op.op_type.as_str(),
                "layerNormalization" | "batchNormalization" | "instanceNormalization"
            ) {
                // Normalization operations - ONNX requires scale/bias as inputs, not attributes
                // Following Chromium's approach: create default initializers when not provided

                let input_id = op.input_operands[0];
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("normalization input lookup", input_id, Some((op, idx)))
                })?;
                let input_data_type = Self::data_type_code(input_operand.descriptor.data_type);
                let input_shape = input_operand.descriptor.static_or_max_shape();
                let final_output_name = operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );
                let mut node_output_name = final_output_name.clone();

                let mut normalized_input_name = operand_name(graph, input_id);
                let mut normalized_input_shape = input_shape.clone();
                let mut transpose_back_perm: Option<Vec<i64>> = None;
                let mut reshape_back_shape: Option<Vec<i64>> = None;
                let mut layernorm_axis_override: Option<i64> = None;

                if op.op_type == "batchNormalization" {
                    let rank = input_shape.len();
                    if rank > 0 {
                        let axis = op
                            .attributes
                            .get("axis")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(1);
                        let normalized_axis = if axis < 0 {
                            (rank as i64 + axis).max(0) as usize
                        } else {
                            axis as usize
                        }
                        .min(rank.saturating_sub(1));

                        if rank == 1 {
                            let channels = *input_shape.first().unwrap_or(&1);
                            normalized_input_name = Self::create_reshape_node(
                                &format!("{}_bn_rank1_to_rank2", op_name),
                                normalized_input_name,
                                vec![1, channels as i64],
                                &mut nodes,
                                &mut initializers,
                            );
                            normalized_input_shape = vec![1, channels];
                            node_output_name = format!("{}_bn_output", op_name);
                            reshape_back_shape = Some(vec![channels as i64]);
                        } else if normalized_axis != 1 {
                            let mut perm: Vec<i64> = (0..rank as i64).collect();
                            perm.swap(1, normalized_axis);
                            let transposed_input_name = format!("{}_bn_axis_to_channel", op_name);
                            nodes.push(NodeProto {
                                input: vec![normalized_input_name],
                                output: vec![transposed_input_name.clone()],
                                name: format!("{}_bn_pre_transpose", op_name),
                                op_type: "Transpose".to_string(),
                                attribute: vec![AttributeProto {
                                    name: "perm".to_string(),
                                    r#type: AttributeType::Ints as i32,
                                    ints: perm.clone(),
                                    ..Default::default()
                                }],
                                ..Default::default()
                            });
                            normalized_input_name = transposed_input_name;
                            normalized_input_shape =
                                perm.iter().map(|&i| input_shape[i as usize]).collect();
                            node_output_name = format!("{}_bn_output", op_name);
                            let mut inverse = vec![0i64; rank];
                            for (new_pos, &old_pos) in perm.iter().enumerate() {
                                inverse[old_pos as usize] = new_pos as i64;
                            }
                            transpose_back_perm = Some(inverse);
                        }
                    }
                }

                if op.op_type == "instanceNormalization" {
                    let rank = input_shape.len();
                    let layout = op
                        .attributes
                        .get("layout")
                        .and_then(|v| v.as_str())
                        .unwrap_or("nchw")
                        .to_ascii_lowercase();
                    if layout == "nhwc" && rank == 4 {
                        let perm = vec![0, 3, 1, 2];
                        let transposed_input_name = format!("{}_in_nchw", op_name);
                        nodes.push(NodeProto {
                            input: vec![normalized_input_name],
                            output: vec![transposed_input_name.clone()],
                            name: format!("{}_in_pre_transpose", op_name),
                            op_type: "Transpose".to_string(),
                            attribute: vec![AttributeProto {
                                name: "perm".to_string(),
                                r#type: AttributeType::Ints as i32,
                                ints: perm.clone(),
                                ..Default::default()
                            }],
                            ..Default::default()
                        });
                        normalized_input_name = transposed_input_name;
                        normalized_input_shape =
                            perm.iter().map(|&i| input_shape[i as usize]).collect();
                        node_output_name = format!("{}_instancenorm_output", op_name);
                        transpose_back_perm = Some(vec![0, 2, 3, 1]);
                    }
                }

                let mut inputs: Vec<String> = vec![normalized_input_name.clone()];

                // Check if scale and bias are provided via attributes (WebNN camelCase)
                let has_scale = op
                    .attributes
                    .get("hasScale")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let has_bias = op
                    .attributes
                    .get("hasBias")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                if op.op_type == "layerNormalization" {
                    let rank = input_shape.len();
                    let axes_raw = op.attributes.get("axes").and_then(|v| v.as_array());
                    if rank == 0 {
                        let output_name = operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        );
                        if has_bias {
                            let mut optional_input_index = 1usize;
                            if has_scale && op.input_operands.len() > optional_input_index {
                                optional_input_index += 1;
                            }
                            if op.input_operands.len() > optional_input_index {
                                nodes.push(NodeProto {
                                    input: vec![operand_name(
                                        graph,
                                        op.input_operands[optional_input_index],
                                    )],
                                    output: vec![output_name],
                                    name: op_name.clone(),
                                    op_type: "Identity".to_string(),
                                    ..Default::default()
                                });
                            } else {
                                let zero_name = format!("{}_zero", op_name);
                                initializers.push(Self::create_scalar_initializer(
                                    zero_name.clone(),
                                    input_data_type,
                                    0.0,
                                ));
                                nodes.push(NodeProto {
                                    input: vec![zero_name],
                                    output: vec![output_name],
                                    name: op_name.clone(),
                                    op_type: "Identity".to_string(),
                                    ..Default::default()
                                });
                            }
                        } else {
                            nodes.push(NodeProto {
                                input: vec![
                                    operand_name(graph, input_id),
                                    operand_name(graph, input_id),
                                ],
                                output: vec![output_name],
                                name: op_name.clone(),
                                op_type: "Sub".to_string(),
                                ..Default::default()
                            });
                        }
                        continue;
                    }

                    if let Some(arr) = axes_raw
                        && arr.is_empty()
                    {
                        let output_name = operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        );
                        if has_bias {
                            let mut optional_input_index = 1usize;
                            if has_scale && op.input_operands.len() > optional_input_index {
                                optional_input_index += 1;
                            }
                            if op.input_operands.len() > optional_input_index {
                                let bias_name =
                                    operand_name(graph, op.input_operands[optional_input_index]);
                                let input_name = operand_name(graph, input_id);
                                let zero_like_name = format!("{}_zero_like", op_name);
                                nodes.push(NodeProto {
                                    input: vec![input_name.clone(), input_name],
                                    output: vec![zero_like_name.clone()],
                                    name: format!("{}_zero_like_sub", op_name),
                                    op_type: "Sub".to_string(),
                                    ..Default::default()
                                });
                                nodes.push(NodeProto {
                                    input: vec![zero_like_name, bias_name],
                                    output: vec![output_name],
                                    name: op_name.clone(),
                                    op_type: "Add".to_string(),
                                    ..Default::default()
                                });
                            } else {
                                let zero_name = format!("{}_zero", op_name);
                                initializers.push(Self::create_scalar_initializer(
                                    zero_name.clone(),
                                    input_data_type,
                                    0.0,
                                ));
                                nodes.push(NodeProto {
                                    input: vec![zero_name],
                                    output: vec![output_name],
                                    name: op_name.clone(),
                                    op_type: "Identity".to_string(),
                                    ..Default::default()
                                });
                            }
                        } else {
                            let input_name = operand_name(graph, input_id);
                            nodes.push(NodeProto {
                                input: vec![input_name.clone(), input_name],
                                output: vec![output_name],
                                name: op_name.clone(),
                                op_type: "Sub".to_string(),
                                ..Default::default()
                            });
                        }
                        continue;
                    }

                    let mut axes = if let Some(arr) = axes_raw {
                        arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>()
                    } else {
                        (1..rank as i64).collect::<Vec<i64>>()
                    };
                    for axis in &mut axes {
                        if *axis < 0 {
                            *axis += rank as i64;
                        }
                    }
                    axes.retain(|&axis| axis >= 0 && axis < rank as i64);
                    let mut seen = std::collections::HashSet::new();
                    axes.retain(|axis| seen.insert(*axis));

                    if !axes.is_empty() {
                        let non_axes: Vec<i64> =
                            (0..rank as i64).filter(|idx| !axes.contains(idx)).collect();
                        let mut perm = non_axes.clone();
                        perm.extend_from_slice(&axes);
                        let identity: Vec<i64> = (0..rank as i64).collect();
                        if perm != identity {
                            let transposed_input_name = format!("{}_ln_axes_tail", op_name);
                            nodes.push(NodeProto {
                                input: vec![normalized_input_name],
                                output: vec![transposed_input_name.clone()],
                                name: format!("{}_ln_pre_transpose", op_name),
                                op_type: "Transpose".to_string(),
                                attribute: vec![AttributeProto {
                                    name: "perm".to_string(),
                                    r#type: AttributeType::Ints as i32,
                                    ints: perm.clone(),
                                    ..Default::default()
                                }],
                                ..Default::default()
                            });
                            normalized_input_name = transposed_input_name;
                            normalized_input_shape =
                                perm.iter().map(|&i| input_shape[i as usize]).collect();
                            node_output_name = format!("{}_ln_output", op_name);
                            let mut inverse = vec![0i64; rank];
                            for (new_pos, &old_pos) in perm.iter().enumerate() {
                                inverse[old_pos as usize] = new_pos as i64;
                            }
                            transpose_back_perm = Some(inverse);
                        }
                    }
                    layernorm_axis_override = Some((rank.saturating_sub(axes.len())) as i64);
                    inputs[0] = normalized_input_name.clone();
                }

                // Determine scale/bias shape based on normalization type
                let mut norm_dynamic_defaults_shape: Option<String> = None;
                let scale_bias_shape = if op.op_type == "layerNormalization" {
                    // For layer norm, ONNX LayerNormalization expects scale/bias shape to match
                    // X.shape[axis:], i.e., all dimensions from axis to the end
                    let axes = op
                        .attributes
                        .get("axes")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                        .unwrap_or_else(|| vec![-1]);

                    // Get the first axis (ONNX only supports a single axis parameter)
                    let first_axis = layernorm_axis_override
                        .or_else(|| axes.first().copied())
                        .unwrap_or(-1);
                    let actual_axis = if first_axis < 0 {
                        ((normalized_input_shape.len() as i64 + first_axis) as usize)
                            .min(normalized_input_shape.len())
                    } else {
                        (first_axis as usize).min(normalized_input_shape.len())
                    };

                    // ONNX LayerNormalization expects scale/bias to match X.shape[axis:].
                    // Keep the tail dimensions, rather than flattening them to a single size.
                    let norm_shape: Vec<i64> = normalized_input_shape
                        .iter()
                        .skip(actual_axis)
                        .map(|d| *d as i64)
                        .collect();

                    // Build runtime layer shape vector when normalized dimensions are dynamic.
                    let has_dynamic_norm_dims = input_operand
                        .descriptor
                        .shape
                        .iter()
                        .skip(actual_axis)
                        .any(|d| matches!(d, Dimension::Dynamic(_)));
                    if has_dynamic_norm_dims {
                        let shape_vec = Self::build_norm_layer_shape_vector(
                            &format!("{}_norm_shape", op_name),
                            operand_name(graph, input_id),
                            actual_axis,
                            input_operand.descriptor.shape.len(),
                            &mut nodes,
                            &mut initializers,
                        );
                        norm_dynamic_defaults_shape = Some(shape_vec);
                    }
                    if norm_shape.is_empty() {
                        vec![1]
                    } else {
                        norm_shape
                    }
                } else if op.op_type == "batchNormalization" {
                    // ONNX BatchNormalization uses channel axis=1 after pre-processing.
                    vec![normalized_input_shape.get(1).copied().unwrap_or(1) as i64]
                } else if op.op_type == "instanceNormalization" {
                    vec![normalized_input_shape.get(1).copied().unwrap_or(1) as i64]
                } else {
                    vec![1]
                };

                // Batch normalization has different input order than layer/instance normalization
                // Python API order: [input, mean, variance, scale?, bias?]
                // ONNX order: [input, scale, bias, mean, variance]
                if op.op_type == "batchNormalization" {
                    // Python API order: [input, mean, variance, scale?, bias?]
                    let mut optional_input_index = 3usize;
                    let scale_input_id =
                        if has_scale && op.input_operands.len() > optional_input_index {
                            let id = op.input_operands[optional_input_index];
                            optional_input_index += 1;
                            Some(id)
                        } else {
                            None
                        };
                    let bias_input_id =
                        if has_bias && op.input_operands.len() > optional_input_index {
                            Some(op.input_operands[optional_input_index])
                        } else {
                            None
                        };

                    if let Some(scale_input_id) = scale_input_id {
                        inputs.push(operand_name(graph, scale_input_id));
                    } else {
                        let scale_name = if let Some(shape_vec) = &norm_dynamic_defaults_shape {
                            Self::create_runtime_filled_tensor(
                                &format!("{}_scale_default", op_name),
                                1.0,
                                input_data_type,
                                shape_vec.clone(),
                                &mut nodes,
                                &mut initializers,
                            )
                        } else {
                            let scale_name = format!("{}_scale_default", op_name);
                            initializers.push(Self::create_vector_initializer(
                                scale_name.clone(),
                                input_data_type,
                                scale_bias_shape.clone(),
                                1.0,
                            ));
                            scale_name
                        };
                        inputs.push(scale_name);
                    }

                    if let Some(bias_input_id) = bias_input_id {
                        inputs.push(operand_name(graph, bias_input_id));
                    } else {
                        let bias_name = if let Some(shape_vec) = &norm_dynamic_defaults_shape {
                            Self::create_runtime_filled_tensor(
                                &format!("{}_bias_default", op_name),
                                0.0,
                                input_data_type,
                                shape_vec.clone(),
                                &mut nodes,
                                &mut initializers,
                            )
                        } else {
                            let bias_name = format!("{}_bias_default", op_name);
                            initializers.push(Self::create_vector_initializer(
                                bias_name.clone(),
                                input_data_type,
                                scale_bias_shape.clone(),
                                0.0,
                            ));
                            bias_name
                        };
                        inputs.push(bias_name);
                    }

                    // Add mean (index 1 - required)
                    if op.input_operands.len() > 1 {
                        inputs.push(operand_name(graph, op.input_operands[1]));
                    }

                    // Add variance (index 2 - required)
                    if op.input_operands.len() > 2 {
                        inputs.push(operand_name(graph, op.input_operands[2]));
                    }
                } else {
                    // Layer normalization and instance normalization
                    // Python API order: [input, scale?, bias?]
                    // ONNX order: [input, scale, bias]
                    let mut optional_input_index = 1usize;
                    let scale_input_id =
                        if has_scale && op.input_operands.len() > optional_input_index {
                            let id = op.input_operands[optional_input_index];
                            optional_input_index += 1;
                            Some(id)
                        } else {
                            None
                        };
                    let bias_input_id =
                        if has_bias && op.input_operands.len() > optional_input_index {
                            Some(op.input_operands[optional_input_index])
                        } else {
                            None
                        };

                    // Add scale input (from operand or create default with 1.0)
                    if let Some(scale_input_id) = scale_input_id {
                        inputs.push(operand_name(graph, scale_input_id));
                    } else {
                        let scale_name = if let Some(shape_vec) = &norm_dynamic_defaults_shape {
                            Self::create_runtime_filled_tensor(
                                &format!("{}_scale_default", op_name),
                                1.0,
                                input_data_type,
                                shape_vec.clone(),
                                &mut nodes,
                                &mut initializers,
                            )
                        } else {
                            let scale_name = format!("{}_scale_default", op_name);
                            initializers.push(Self::create_vector_initializer(
                                scale_name.clone(),
                                input_data_type,
                                scale_bias_shape.clone(),
                                1.0,
                            ));
                            scale_name
                        };
                        inputs.push(scale_name);
                    }

                    // Add bias input (from operand or create default with 0.0) - optional for layer norm
                    if let Some(bias_input_id) = bias_input_id {
                        inputs.push(operand_name(graph, bias_input_id));
                    } else if op.op_type != "layerNormalization" || has_bias {
                        let bias_name = if let Some(shape_vec) = &norm_dynamic_defaults_shape {
                            Self::create_runtime_filled_tensor(
                                &format!("{}_bias_default", op_name),
                                0.0,
                                input_data_type,
                                shape_vec.clone(),
                                &mut nodes,
                                &mut initializers,
                            )
                        } else {
                            let bias_name = format!("{}_bias_default", op_name);
                            initializers.push(Self::create_vector_initializer(
                                bias_name.clone(),
                                input_data_type,
                                scale_bias_shape.clone(),
                                0.0,
                            ));
                            bias_name
                        };
                        inputs.push(bias_name);
                    }
                }

                let mut attributes = Self::create_operation_attributes(op);
                if op.op_type == "layerNormalization"
                    && let Some(axis) = layernorm_axis_override
                    && let Some(attr) = attributes.iter_mut().find(|a| a.name == "axis")
                {
                    attr.i = axis;
                }
                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![node_output_name.clone()],
                    name: op_name.clone(),
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: attributes,
                    ..Default::default()
                });

                if let Some(perm) = transpose_back_perm {
                    let transpose_back_output = if reshape_back_shape.is_some() {
                        format!("{}_norm_transposed_back", op_name)
                    } else {
                        final_output_name.clone()
                    };
                    nodes.push(NodeProto {
                        input: vec![node_output_name],
                        output: vec![transpose_back_output.clone()],
                        name: format!("{}_norm_post_transpose", op_name),
                        op_type: "Transpose".to_string(),
                        attribute: vec![AttributeProto {
                            name: "perm".to_string(),
                            r#type: AttributeType::Ints as i32,
                            ints: perm,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    node_output_name = transpose_back_output;
                }

                if let Some(shape) = reshape_back_shape {
                    let reshaped = Self::create_reshape_node(
                        &format!("{}_norm_post_reshape", op_name),
                        node_output_name,
                        shape,
                        &mut nodes,
                        &mut initializers,
                    );
                    nodes.push(NodeProto {
                        input: vec![reshaped],
                        output: vec![final_output_name],
                        name: format!("{}_norm_post_identity", op_name),
                        op_type: "Identity".to_string(),
                        ..Default::default()
                    });
                }
            } else if op.op_type == "hardSwish" {
                // HardSwish decomposition: x * clip(x + 3, 0, 6) / 6
                // ONNX opset 13 doesn't have HardSwish, so we decompose it
                let input_name = operand_name(graph, op.input_operands[0]);
                let output_name = operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );

                // Get input data type for scalar initializers
                let input_operand = graph.operand(op.input_operands[0]).ok_or_else(|| {
                    Self::invalid_operand(
                        "hardSwish input lookup",
                        op.input_operands[0],
                        Some((op, idx)),
                    )
                })?;
                let dtype = Self::data_type_code(input_operand.descriptor.data_type);

                // Step 1: Add 3 to x
                let three_name = format!("{}_const_3", op_name);
                initializers.push(Self::create_scalar_initializer(
                    three_name.clone(),
                    dtype,
                    3.0,
                ));

                let add_output = format!("{}_add_3", op_name);
                nodes.push(NodeProto {
                    input: vec![input_name.clone(), three_name],
                    output: vec![add_output.clone()],
                    name: format!("{}_add", op_name),
                    op_type: "Add".to_string(),
                    ..Default::default()
                });

                // Step 2: Clip to [0, 6]
                let zero_name = format!("{}_const_0", op_name);
                let six_name = format!("{}_const_6", op_name);
                initializers.push(Self::create_scalar_initializer(
                    zero_name.clone(),
                    dtype,
                    0.0,
                ));
                initializers.push(Self::create_scalar_initializer(
                    six_name.clone(),
                    dtype,
                    6.0,
                ));

                let clip_output = format!("{}_clip", op_name);
                nodes.push(NodeProto {
                    input: vec![add_output, zero_name, six_name],
                    output: vec![clip_output.clone()],
                    name: format!("{}_clip", op_name),
                    op_type: "Clip".to_string(),
                    ..Default::default()
                });

                // Step 3: Divide by 6
                let six_div_name = format!("{}_const_6_div", op_name);
                initializers.push(Self::create_scalar_initializer(
                    six_div_name.clone(),
                    dtype,
                    6.0,
                ));

                let div_output = format!("{}_div", op_name);
                nodes.push(NodeProto {
                    input: vec![clip_output, six_div_name],
                    output: vec![div_output.clone()],
                    name: format!("{}_div", op_name),
                    op_type: "Div".to_string(),
                    ..Default::default()
                });

                // Step 4: Multiply by x
                nodes.push(NodeProto {
                    input: vec![input_name, div_output],
                    output: vec![output_name],
                    name: format!("{}_mul", op_name),
                    op_type: "Mul".to_string(),
                    ..Default::default()
                });
            } else if op.op_type == "unsqueeze" || op.op_type == "squeeze" {
                // Unsqueeze/Squeeze operations - in ONNX opset 13+, axes must be provided as input tensor
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Get axes from attributes and create as input tensor
                if let Some(axes_i64) = Self::parse_i64_array(op, "axes") {
                    let axes_name = format!("{}_axes", op_name);
                    inputs.push(axes_name.clone());

                    initializers.push(TensorProto {
                        name: axes_name,
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![axes_i64.len() as i64],
                        int64_data: axes_i64,
                        ..Default::default()
                    });
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: vec![], // No attributes for Unsqueeze/Squeeze in opset 13+
                    ..Default::default()
                });
            } else if op.op_type.eq_ignore_ascii_case("concat") {
                let mut concat_inputs: Vec<String> = Vec::new();
                let input_ranks: Vec<usize> = op
                    .input_operands
                    .iter()
                    .map(|id| {
                        operand_shapes
                            .get(id)
                            .map(|shape| shape.len())
                            .or_else(|| {
                                graph
                                    .operand(*id)
                                    .map(|operand| operand.descriptor.shape.len())
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                let has_rank_mismatch = input_ranks
                    .windows(2)
                    .any(|w| w.first().copied().unwrap_or(0) != w.get(1).copied().unwrap_or(0));
                let output_rank_is_1d = op
                    .output_operand
                    .and_then(|id| graph.operand(id))
                    .map(|operand| operand.descriptor.shape.len() == 1)
                    .unwrap_or(false);

                for (input_idx, input_id) in op.input_operands.iter().enumerate() {
                    let base_name = operand_name(graph, *input_id);
                    let rank = input_ranks.get(input_idx).copied().unwrap_or(0);
                    if (has_rank_mismatch || output_rank_is_1d) && rank > 1 {
                        let shape_name = format!("{}_flatten_{}_shape", op_name, input_idx);
                        initializers.push(TensorProto {
                            name: shape_name.clone(),
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![1],
                            int64_data: vec![-1],
                            ..Default::default()
                        });
                        let flat_name = format!("{}_flatten_{}", op_name, input_idx);
                        nodes.push(NodeProto {
                            input: vec![base_name, shape_name],
                            output: vec![flat_name.clone()],
                            name: format!("{}_reshape_flatten_{}", op_name, input_idx),
                            op_type: "Reshape".to_string(),
                            attribute: vec![],
                            ..Default::default()
                        });
                        concat_inputs.push(flat_name);
                    } else {
                        concat_inputs.push(base_name);
                    }
                }

                nodes.push(NodeProto {
                    input: concat_inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: Self::create_operation_attributes(op),
                    ..Default::default()
                });
            } else if op.op_type.eq_ignore_ascii_case("scatternd") {
                // ScatterND requires int64 indices - insert Cast if needed
                let mut inputs: Vec<String> = Vec::new();

                // Debug: print shapes of all inputs
                debug_print!("[SCATTERND DEBUG] Operation: {}", op_name);
                for (i, &input_id) in op.input_operands.iter().enumerate() {
                    let shape = operand_shapes.get(&input_id);
                    let desc_shape = graph.operand(input_id).map(|o| &o.descriptor.shape);
                    debug_print!(
                        "  Input {}: operand_id={}, tracked_shape={:?}, descriptor_shape={:?}",
                        i,
                        input_id,
                        shape,
                        desc_shape
                    );
                }

                // Input 0: data
                if let Some(&data_id) = op.input_operands.first() {
                    inputs.push(operand_name(graph, data_id));
                }

                // Input 1: indices - must be int64
                if let Some(&indices_id) = op.input_operands.get(1) {
                    let data_shape = op
                        .input_operands
                        .first()
                        .and_then(|id| operand_shapes.get(id))
                        .cloned()
                        .or_else(|| {
                            op.input_operands
                                .first()
                                .and_then(|id| graph.operand(*id))
                                .map(|o| o.descriptor.static_or_max_shape())
                        })
                        .unwrap_or_default();
                    if let Some(indices_operand) = graph.operand(indices_id) {
                        let indices_dtype = type_overrides
                            .get(&indices_id)
                            .copied()
                            .unwrap_or(indices_operand.descriptor.data_type);

                        let mut indices_for_scatter = if !matches!(indices_dtype, DataType::Int64) {
                            // Insert Cast node to convert indices to int64
                            let cast_output = format!("{}_indices_cast", op_name);
                            nodes.push(NodeProto {
                                input: vec![operand_name(graph, indices_id)],
                                output: vec![cast_output.clone()],
                                name: format!("{}_cast_indices", op_name),
                                op_type: "Cast".to_string(),
                                attribute: vec![AttributeProto {
                                    name: "to".to_string(),
                                    r#type: AttributeType::Int as i32,
                                    i: ProtoDataType::Int64 as i32 as i64,
                                    ..Default::default()
                                }],
                                ..Default::default()
                            });
                            cast_output
                        } else {
                            operand_name(graph, indices_id)
                        };

                        // Clamp indices to valid bounds per index component.
                        let k = indices_operand
                            .descriptor
                            .shape
                            .last()
                            .map(get_static_or_max_size)
                            .unwrap_or(1) as usize;
                        if k > 0 && k <= data_shape.len() {
                            let mins: Vec<i64> =
                                data_shape[..k].iter().map(|&d| -(d as i64)).collect();
                            let maxs: Vec<i64> =
                                data_shape[..k].iter().map(|&d| d as i64 - 1).collect();

                            let min_name = format!("{}_scatter_indices_min", op_name);
                            let max_name = format!("{}_scatter_indices_max", op_name);
                            let low_name = format!("{}_scatter_indices_low", op_name);
                            let clamped_name = format!("{}_scatter_indices_clamped", op_name);
                            initializers.push(TensorProto {
                                name: min_name.clone(),
                                data_type: ProtoDataType::Int64 as i32,
                                dims: vec![k as i64],
                                int64_data: mins,
                                ..Default::default()
                            });
                            initializers.push(TensorProto {
                                name: max_name.clone(),
                                data_type: ProtoDataType::Int64 as i32,
                                dims: vec![k as i64],
                                int64_data: maxs,
                                ..Default::default()
                            });
                            nodes.push(NodeProto {
                                input: vec![indices_for_scatter, min_name],
                                output: vec![low_name.clone()],
                                name: format!("{}_scatter_indices_max", op_name),
                                op_type: "Max".to_string(),
                                ..Default::default()
                            });
                            nodes.push(NodeProto {
                                input: vec![low_name, max_name],
                                output: vec![clamped_name.clone()],
                                name: format!("{}_scatter_indices_min", op_name),
                                op_type: "Min".to_string(),
                                ..Default::default()
                            });
                            indices_for_scatter = clamped_name;
                        }

                        inputs.push(indices_for_scatter);
                    } else {
                        inputs.push(operand_name(graph, indices_id));
                    }
                }

                // Input 2: updates
                if let Some(&updates_id) = op.input_operands.get(2) {
                    inputs.push(operand_name(graph, updates_id));
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: "ScatterND".to_string(),
                    attribute: vec![],
                    ..Default::default()
                });
            } else {
                if op.op_type.eq_ignore_ascii_case("where") {
                    let mut inputs: Vec<String> = Vec::with_capacity(3);
                    let cond_id = op.input_operands[0];
                    let true_id = op.input_operands[1];
                    let false_id = op.input_operands[2];

                    inputs.push(operand_name(graph, cond_id));

                    let true_type = graph
                        .operand(true_id)
                        .map(|operand| {
                            type_overrides
                                .get(&true_id)
                                .copied()
                                .unwrap_or(operand.descriptor.data_type)
                        })
                        .ok_or_else(|| {
                            Self::invalid_operand("where true input", true_id, Some((op, idx)))
                        })?;
                    if graph.operand(false_id).is_none() {
                        return Err(Self::invalid_operand(
                            "where false input",
                            false_id,
                            Some((op, idx)),
                        ));
                    }

                    let target_type = true_type;

                    let true_input_name = operand_name(graph, true_id);
                    let true_cast_output_name = format!("{}_true_cast_{}", op_name, cast_counter);
                    cast_counter += 1;
                    nodes.push(Self::create_cast_node(
                        &format!("{}_cast_true_{}", op_name, cast_counter),
                        true_input_name,
                        true_cast_output_name.clone(),
                        Self::data_type_code(target_type),
                    ));
                    inputs.push(true_cast_output_name);

                    let false_input_name = operand_name(graph, false_id);
                    let false_cast_output_name = format!("{}_false_cast_{}", op_name, cast_counter);
                    cast_counter += 1;
                    nodes.push(Self::create_cast_node(
                        &format!("{}_cast_false_{}", op_name, cast_counter),
                        false_input_name,
                        false_cast_output_name.clone(),
                        Self::data_type_code(target_type),
                    ));
                    inputs.push(false_cast_output_name);

                    let output_operand_id =
                        op.output_operand.expect("Single-output operation expected");
                    type_overrides.insert(output_operand_id, target_type);

                    nodes.push(NodeProto {
                        input: inputs,
                        output: vec![operand_name(graph, output_operand_id)],
                        name: op_name,
                        op_type: Self::onnx_op_type(&op.op_type),
                        attribute: Self::create_operation_attributes(op),
                        ..Default::default()
                    });
                    continue;
                }

                // Check if operation requires float types (ONNX limitation)
                let has_float_inputs = op.input_operands.iter().any(|&input_id| {
                    graph
                        .operand(input_id)
                        .map(|operand| {
                            let dtype = type_overrides
                                .get(&input_id)
                                .copied()
                                .unwrap_or(operand.descriptor.data_type);
                            matches!(dtype, DataType::Float32 | DataType::Float16)
                        })
                        .unwrap_or(false)
                });
                let requires_float = matches!(
                    op.op_type.as_str(),
                    "relu"
                        | "sigmoid"
                        | "tanh"
                        | "softmax"
                        | "elu"
                        | "leakyRelu"
                        | "prelu"
                        | "hardSigmoid"
                        | "hardSwish"
                        | "softplus"
                        | "softsign"
                        | "sub"
                        | "div"
                        | "pow"
                );

                // Check if any inputs have integer types
                let has_integer_inputs = op.input_operands.iter().any(|&input_id| {
                    if let Some(operand) = graph.operand(input_id) {
                        let dtype = type_overrides
                            .get(&input_id)
                            .copied()
                            .unwrap_or(operand.descriptor.data_type);
                        matches!(
                            dtype,
                            DataType::Int8
                                | DataType::Uint8
                                | DataType::Int32
                                | DataType::Uint32
                                | DataType::Int64
                                | DataType::Uint64
                        )
                    } else {
                        false
                    }
                });

                let mixed_numeric_inputs = has_integer_inputs
                    && has_float_inputs
                    && matches!(op.op_type.as_str(), "mul" | "add");

                if (requires_float && has_integer_inputs) || mixed_numeric_inputs {
                    // Cast inputs to float32, execute operation, cast output back
                    let mut cast_inputs = Vec::new();
                    let mut original_types = Vec::new();

                    for &input_id in &op.input_operands {
                        let input_name = operand_name(graph, input_id);
                        let input_operand = graph.operand(input_id).ok_or_else(|| {
                            Self::invalid_operand(
                                "float-cast input lookup",
                                input_id,
                                Some((op, idx)),
                            )
                        })?;

                        let dtype = type_overrides
                            .get(&input_id)
                            .copied()
                            .unwrap_or(input_operand.descriptor.data_type);
                        original_types.push(dtype);

                        if matches!(
                            input_operand.descriptor.data_type,
                            DataType::Int8
                                | DataType::Uint8
                                | DataType::Int32
                                | DataType::Uint32
                                | DataType::Int64
                                | DataType::Uint64
                        ) {
                            // Cast to float32
                            let cast_output_name =
                                format!("{}_input_{}_float32", op_name, cast_counter);
                            cast_counter += 1;

                            nodes.push(Self::create_cast_node(
                                &format!("cast_to_float32_{}", cast_counter - 1),
                                input_name,
                                cast_output_name.clone(),
                                ProtoDataType::Float,
                            ));

                            cast_inputs.push(cast_output_name);
                        } else {
                            cast_inputs.push(input_name);
                        }
                    }

                    // Create the operation node (outputs float32)
                    let float_output_name = format!("{}_float32_output", op_name);
                    let output_operand_id =
                        op.output_operand.expect("Single-output operation expected");
                    let final_output_name = operand_name(graph, output_operand_id);
                    let op_output_name = if requires_float && !mixed_numeric_inputs {
                        float_output_name.clone()
                    } else {
                        final_output_name.clone()
                    };
                    let attributes = Self::create_operation_attributes(op);

                    nodes.push(NodeProto {
                        input: cast_inputs,
                        output: vec![op_output_name.clone()],
                        name: op_name.clone(),
                        op_type: Self::onnx_op_type(&op.op_type),
                        attribute: attributes,
                        ..Default::default()
                    });

                    // Cast output back to original type (use first input's type as reference)
                    if requires_float && !mixed_numeric_inputs {
                        let output_type = original_types[0];
                        nodes.push(Self::create_cast_node(
                            &format!("{}_cast_output", op_name),
                            float_output_name,
                            final_output_name,
                            Self::data_type_code(output_type),
                        ));
                        type_overrides.insert(output_operand_id, output_type);
                    } else {
                        // Keep float output for mixed numeric inputs so downstream ops see a float
                        type_overrides.insert(output_operand_id, DataType::Float32);
                    }
                } else {
                    // Regular operation - no Cast nodes needed
                    let attributes = Self::create_operation_attributes(op);

                    nodes.push(NodeProto {
                        input: op
                            .input_operands
                            .iter()
                            .map(|id| {
                                // Resolve remapping for skipped concat outputs
                                let resolved_id = operand_remapping.get(id).copied().unwrap_or(*id);
                                operand_name(graph, resolved_id)
                            })
                            .collect(),
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: op_name,
                        op_type: Self::onnx_op_type(&op.op_type),
                        attribute: attributes,
                        ..Default::default()
                    });
                }
            }
        }

        // Add value_info ONLY for operands where we have explicit shape/type tracking
        // This provides guidance to ONNX Runtime for operations we explicitly handle
        // (concat, unsqueeze, binary ops) while letting it infer shapes for others
        let mut seen_names = std::collections::HashSet::new();
        for vi in inputs_val.iter().chain(outputs_val.iter()) {
            if !vi.name.is_empty() {
                seen_names.insert(vi.name.clone());
            }
        }

        // Do not emit intermediate value_info entries. Imported ONNX graphs can carry
        // descriptor dtypes that diverge from ONNX's inferred tensor types on shape-only
        // ops (e.g., Unsqueeze/Expand), which causes model load failures. Inputs/outputs
        // remain strongly typed via graph.input/graph.output.
        let _ = (
            &unsqueeze_like_outputs,
            &shape_overrides,
            &type_overrides,
            &seen_names,
        );

        let graph_proto = GraphProto {
            name: "webnn_graph".to_string(),
            node: nodes,
            input: inputs_val,
            output: outputs_val,
            initializer: initializers,
            value_info: value_infos,
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: 8, // IR version 8 = ONNX 1.10+ (supports opset 14-15)
            model_version: 1,
            producer_name: "rustnn".to_string(),
            producer_version: "0.1.0".to_string(),
            graph: Some(graph_proto),
            opset_import: vec![OperatorSetIdProto {
                version: 14,            // Opset 14 adds Trilu support
                domain: "".to_string(), // Empty string = default ONNX domain
            }],
            ..Default::default()
        };

        let data = model.encode_to_vec();

        Ok(ConvertedGraph {
            format: "onnx",
            content_type: "application/onnx",
            data,
            weights_data: None, // ONNX doesn't use external weight files
        })
    }
}

fn value_info(name: &str, desc: &crate::graph::OperandDescriptor) -> ValueInfoProto {
    let dims = desc.shape.iter().map(|d| match d {
        crate::graph::Dimension::Static(v) => crate::protos::onnx::tensor_shape_proto::Dimension {
            value: Some(
                crate::protos::onnx::tensor_shape_proto::dimension::Value::DimValue(*v as i64),
            ),
            ..Default::default()
        },
        crate::graph::Dimension::Dynamic(dd) => {
            crate::protos::onnx::tensor_shape_proto::Dimension {
                value: Some(
                    crate::protos::onnx::tensor_shape_proto::dimension::Value::DimParam(
                        dd.name.clone(),
                    ),
                ),
                ..Default::default()
            }
        }
    });
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(crate::protos::onnx::type_proto::Value::TensorType(
                TensorTypeProto {
                    elem_type: OnnxConverter::data_type_code(desc.data_type) as i32,
                    shape: Some(TensorShapeProto {
                        dim: dims.collect(),
                    }),
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::GraphConverter;
    use crate::graph::{
        DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
        Operation,
    };
    use crate::protos::onnx::tensor_proto::DataType as ProtoDataType;
    use std::collections::HashMap;

    fn s(shape: &[u32]) -> Vec<Dimension> {
        crate::graph::to_dimension_vector(shape)
    }

    #[test]
    fn test_data_type_code_int4() {
        let code = OnnxConverter::data_type_code(DataType::Int4);
        assert_eq!(code, ProtoDataType::Int32);
    }

    #[test]
    fn test_data_type_code_uint4() {
        let code = OnnxConverter::data_type_code(DataType::Uint4);
        assert_eq!(code, ProtoDataType::Uint8);
    }

    #[test]
    fn test_data_type_code_float32() {
        let code = OnnxConverter::data_type_code(DataType::Float32);
        assert_eq!(code, ProtoDataType::Float);
    }

    #[test]
    fn test_data_type_code_float16() {
        let code = OnnxConverter::data_type_code(DataType::Float16);
        assert_eq!(code, ProtoDataType::Float16);
    }

    #[test]
    fn test_cumulative_sum_op_maps_to_onnx_cumsum() {
        assert_eq!(OnnxConverter::onnx_op_type("cumulativeSum"), "CumSum");
        assert_eq!(OnnxConverter::onnx_op_type("cumulative_sum"), "CumSum");
    }

    #[test]
    fn test_gru_cell_op_maps_to_onnx_gru() {
        assert_eq!(OnnxConverter::onnx_op_type("gruCell"), "GRU");
        assert_eq!(OnnxConverter::onnx_op_type("gru_cell"), "GRU");
    }

    #[test]
    fn test_round_even_op_maps_to_onnx_round() {
        assert_eq!(OnnxConverter::onnx_op_type("roundEven"), "Round");
        assert_eq!(OnnxConverter::onnx_op_type("round_even"), "Round");
    }

    #[test]
    fn test_create_reshape_node() {
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();

        let output_name = OnnxConverter::create_reshape_node(
            "test",
            "input".to_string(),
            vec![2, 3, 4],
            &mut nodes,
            &mut initializers,
        );

        assert_eq!(output_name, "test_reshaped");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].op_type, "Reshape");
        assert_eq!(nodes[0].input.len(), 2);
        assert_eq!(nodes[0].input[0], "input");
        assert_eq!(nodes[0].input[1], "test_shape");
        assert_eq!(nodes[0].output.len(), 1);
        assert_eq!(nodes[0].output[0], "test_reshaped");

        assert_eq!(initializers.len(), 1);
        assert_eq!(initializers[0].name, "test_shape");
        assert_eq!(initializers[0].data_type, ProtoDataType::Int64 as i32);
        assert_eq!(initializers[0].int64_data, vec![2, 3, 4]);
    }

    #[test]
    fn test_create_reshape_node_to_scalar() {
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();

        let output_name = OnnxConverter::create_reshape_node(
            "scalar",
            "input".to_string(),
            vec![],
            &mut nodes,
            &mut initializers,
        );

        assert_eq!(output_name, "scalar_reshaped");
        assert_eq!(initializers[0].dims, vec![0]);
        assert_eq!(initializers[0].int64_data, Vec::<i64>::new());
    }

    #[test]
    fn test_quantize_linear_conversion_per_tensor() {
        let mut operands = Vec::new();
        let mut operations = Vec::new();

        operands.push(Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[1, 3, 224, 224]),
                pending_permutation: vec![],
            },
            name: Some("input".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[]),
                pending_permutation: vec![],
            },
            name: Some("scale".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Int8,
                shape: s(&[]),
                pending_permutation: vec![],
            },
            name: Some("zero_point".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Int8,
                shape: s(&[1, 3, 224, 224]),
                pending_permutation: vec![],
            },
            name: Some("output".to_string()),
        });

        operations.push(Operation {
            op_type: "quantizeLinear".to_string(),
            input_operands: vec![0, 1, 2],
            output_operand: Some(3),
            output_operands: vec![],
            attributes: serde_json::json!({}),
            label: None,
        });

        let graph = GraphInfo {
            operands,
            input_operands: vec![0],
            output_operands: vec![3],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        let converter = OnnxConverter;
        let result = converter.convert(&graph);
        assert!(result.is_ok());

        let converted = result.unwrap();
        assert_eq!(converted.format, "onnx");

        let model = ModelProto::decode(converted.data.as_slice()).unwrap();
        let graph_proto = model.graph.unwrap();

        assert!(
            graph_proto
                .node
                .iter()
                .all(|n| n.op_type != "QuantizeLinear")
        );
        let final_node = graph_proto
            .node
            .iter()
            .find(|n| n.output.iter().any(|o| o == "output"));
        assert!(final_node.is_some());
        assert_eq!(final_node.unwrap().op_type, "Cast");
    }

    #[test]
    fn test_dequantize_linear_conversion() {
        let mut operands = Vec::new();
        let mut operations = Vec::new();

        operands.push(Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Int8,
                shape: s(&[1, 64]),
                pending_permutation: vec![],
            },
            name: Some("quantized_input".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[]),
                pending_permutation: vec![],
            },
            name: Some("scale".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Int8,
                shape: s(&[]),
                pending_permutation: vec![],
            },
            name: Some("zero_point".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[1, 64]),
                pending_permutation: vec![],
            },
            name: Some("output".to_string()),
        });

        operations.push(Operation {
            op_type: "dequantizeLinear".to_string(),
            input_operands: vec![0, 1, 2],
            output_operand: Some(3),
            output_operands: vec![],
            attributes: serde_json::json!({}),
            label: None,
        });

        let graph = GraphInfo {
            operands,
            input_operands: vec![0],
            output_operands: vec![3],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        let converter = OnnxConverter;
        let result = converter.convert(&graph);
        assert!(result.is_ok());

        let converted = result.unwrap();
        let model = ModelProto::decode(converted.data.as_slice()).unwrap();
        let graph_proto = model.graph.unwrap();

        assert!(
            graph_proto
                .node
                .iter()
                .all(|n| n.op_type != "DequantizeLinear")
        );
        let final_node = graph_proto
            .node
            .iter()
            .find(|n| n.output.iter().any(|o| o == "output"));
        assert!(final_node.is_some());
        assert_eq!(final_node.unwrap().op_type, "Mul");
    }

    #[test]
    fn test_quantize_linear_per_axis() {
        let mut operands = Vec::new();
        let mut operations = Vec::new();

        operands.push(Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[1, 64, 128]),
                pending_permutation: vec![],
            },
            name: Some("input".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[1, 64, 1]),
                pending_permutation: vec![],
            },
            name: Some("scale".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Int8,
                shape: s(&[1, 64, 1]),
                pending_permutation: vec![],
            },
            name: Some("zero_point".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Int8,
                shape: s(&[1, 64, 128]),
                pending_permutation: vec![],
            },
            name: Some("output".to_string()),
        });

        operations.push(Operation {
            op_type: "quantizeLinear".to_string(),
            input_operands: vec![0, 1, 2],
            output_operand: Some(3),
            output_operands: vec![],
            attributes: serde_json::json!({}),
            label: None,
        });

        let graph = GraphInfo {
            operands,
            input_operands: vec![0],
            output_operands: vec![3],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        let converter = OnnxConverter;
        let result = converter.convert(&graph);
        assert!(result.is_ok());

        let converted = result.unwrap();
        let model = ModelProto::decode(converted.data.as_slice()).unwrap();
        let graph_proto = model.graph.unwrap();

        assert!(
            graph_proto
                .node
                .iter()
                .all(|n| n.op_type != "QuantizeLinear")
        );
        let final_node = graph_proto
            .node
            .iter()
            .find(|n| n.output.iter().any(|o| o == "output"));
        assert!(final_node.is_some());
        assert_eq!(final_node.unwrap().op_type, "Cast");
    }

    #[test]
    fn test_int4_constant_byte_length() {
        let operands = vec![Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: s(&[16, 16]),
                pending_permutation: vec![],
            },
            name: Some("weight".to_string()),
        }];

        let graph = GraphInfo {
            operands,
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        let converter = OnnxConverter;
        let result = converter.convert(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_uint4_constant_byte_length() {
        let operands = vec![Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Uint4,
                shape: s(&[32, 32]),
                pending_permutation: vec![],
            },
            name: Some("weight".to_string()),
        }];

        let graph = GraphInfo {
            operands,
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        let converter = OnnxConverter;
        let result = converter.convert(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cast_operation_with_int4_target() {
        let mut operands = Vec::new();
        let mut operations = Vec::new();

        operands.push(Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[10, 10]),
                pending_permutation: vec![],
            },
            name: Some("input".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: s(&[10, 10]),
                pending_permutation: vec![],
            },
            name: Some("output".to_string()),
        });

        operations.push(Operation {
            op_type: "cast".to_string(),
            input_operands: vec![0],
            output_operand: Some(1),
            output_operands: vec![],
            attributes: serde_json::json!({
                "to": "int4"
            }),
            label: None,
        });

        let graph = GraphInfo {
            operands,
            input_operands: vec![0],
            output_operands: vec![1],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converter = OnnxConverter;
        let result = converter.convert(&graph);
        assert!(result.is_ok());
    }

    #[cfg(not(feature = "dynamic-inputs"))]
    #[test]
    fn test_dynamic_dimensions_require_feature_opt_in() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            Dimension::Dynamic(DynamicDimension {
                                name: "batch".to_string(),
                                max_size: 8,
                            }),
                            Dimension::Static(4),
                        ],
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            Dimension::Dynamic(DynamicDimension {
                                name: "batch".to_string(),
                                max_size: 8,
                            }),
                            Dimension::Static(4),
                        ],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation {
                op_type: "identity".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: vec![],
                attributes: serde_json::json!({}),
                label: None,
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let err = OnnxConverter.convert(&graph).unwrap_err();
        assert!(matches!(err, GraphError::DynamicInputsFeatureDisabled));
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn test_dynamic_dim_emits_dim_param() {
        let mut operands = Vec::new();
        let mut operations = Vec::new();

        operands.push(Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![
                    Dimension::Dynamic(DynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    Dimension::Static(4),
                ],
                pending_permutation: vec![],
            },
            name: Some("input".to_string()),
        });

        operands.push(Operand {
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![
                    Dimension::Dynamic(DynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    Dimension::Static(4),
                ],
                pending_permutation: vec![],
            },
            name: Some("output".to_string()),
        });

        operations.push(Operation {
            op_type: "identity".to_string(),
            input_operands: vec![0],
            output_operand: Some(1),
            output_operands: vec![1],
            attributes: serde_json::json!({}),
            label: None,
        });

        let graph = GraphInfo {
            operands,
            input_operands: vec![0],
            output_operands: vec![1],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converter = OnnxConverter;
        let result = converter.convert(&graph);
        assert!(result.is_ok());

        let converted = result.unwrap();
        let model = ModelProto::decode(converted.data.as_slice()).unwrap();
        let graph_proto = model.graph.unwrap();
        let input_vi = &graph_proto.input[0];
        let tensor_type = match input_vi.r#type.as_ref().unwrap().value.as_ref().unwrap() {
            crate::protos::onnx::type_proto::Value::TensorType(t) => t,
            _ => panic!("expected tensor type"),
        };
        let shape = tensor_type.shape.as_ref().unwrap();

        let dim0 = &shape.dim[0];
        let dim1 = &shape.dim[1];

        match dim0.value.as_ref().unwrap() {
            crate::protos::onnx::tensor_shape_proto::dimension::Value::DimParam(p) => {
                assert_eq!(p, "batch");
            }
            _ => panic!("expected dim_param for dynamic batch"),
        }

        match dim1.value.as_ref().unwrap() {
            crate::protos::onnx::tensor_shape_proto::dimension::Value::DimValue(v) => {
                assert_eq!(*v, 4);
            }
            _ => panic!("expected dim_value for static 4"),
        }
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn test_cumulative_sum_lowers_to_cumsum_with_axis_input() {
        let operands = vec![
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![2, 3],
                    pending_permutation: vec![],
                },
                name: Some("input".to_string()),
            },
            Operand {
                kind: OperandKind::Output,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![2, 3],
                    pending_permutation: vec![],
                },
                name: Some("output".to_string()),
            },
        ];

        let operations = vec![Operation {
            op_type: "cumulativeSum".to_string(),
            input_operands: vec![0],
            output_operand: Some(1),
            output_operands: vec![],
            attributes: serde_json::json!({
                "axis": 1,
                "exclusive": true,
                "reversed": true
            }),
            label: None,
        }];

        let graph = GraphInfo {
            operands,
            input_operands: vec![0],
            output_operands: vec![1],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = OnnxConverter
            .convert(&graph)
            .expect("cumulativeSum conversion should succeed");
        let model = ModelProto::decode(converted.data.as_slice()).expect("decode model");
        let graph_proto = model.graph.expect("graph");

        let node = graph_proto
            .node
            .iter()
            .find(|n| n.op_type == "CumSum")
            .expect("CumSum node");
        assert_eq!(node.input.len(), 2);

        let axis_init = graph_proto
            .initializer
            .iter()
            .find(|t| t.name == node.input[1])
            .expect("axis initializer");
        assert_eq!(axis_init.int64_data, vec![1]);

        let exclusive_attr = node
            .attribute
            .iter()
            .find(|a| a.name == "exclusive")
            .expect("exclusive attr");
        assert_eq!(exclusive_attr.i, 1);

        let reverse_attr = node
            .attribute
            .iter()
            .find(|a| a.name == "reverse")
            .expect("reverse attr");
        assert_eq!(reverse_attr.i, 1);
    }

    #[test]
    fn test_gru_cell_lowers_to_gru_with_reordered_bias_for_rzn() {
        let operands = vec![
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![Dimension::Static(3), Dimension::Static(2)],
                    pending_permutation: vec![],
                },
                name: Some("x".to_string()),
            },
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![Dimension::Static(12), Dimension::Static(2)],
                    pending_permutation: vec![],
                },
                name: Some("w".to_string()),
            },
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![Dimension::Static(12), Dimension::Static(4)],
                    pending_permutation: vec![],
                },
                name: Some("r".to_string()),
            },
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![Dimension::Static(3), Dimension::Static(4)],
                    pending_permutation: vec![],
                },
                name: Some("h".to_string()),
            },
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![Dimension::Static(12)],
                    pending_permutation: vec![],
                },
                name: Some("b".to_string()),
            },
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![Dimension::Static(12)],
                    pending_permutation: vec![],
                },
                name: Some("rb".to_string()),
            },
            Operand {
                kind: OperandKind::Output,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![Dimension::Static(3), Dimension::Static(4)],
                    pending_permutation: vec![],
                },
                name: Some("y".to_string()),
            },
        ];

        let operations = vec![Operation {
            op_type: "gruCell".to_string(),
            input_operands: vec![0, 1, 2, 3, 4, 5],
            output_operand: Some(6),
            output_operands: vec![],
            attributes: serde_json::json!({
                "hiddenSize": 4,
                "layout": "rzn",
                "resetAfter": false,
                "activations": ["relu", "relu"]
            }),
            label: None,
        }];

        let graph = GraphInfo {
            operands,
            input_operands: vec![0, 1, 2, 3, 4, 5],
            output_operands: vec![6],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = OnnxConverter
            .convert(&graph)
            .expect("gruCell conversion should succeed");
        let model = ModelProto::decode(converted.data.as_slice()).expect("decode model");
        let graph_proto = model.graph.expect("graph");

        let gru_node = graph_proto
            .node
            .iter()
            .find(|n| n.op_type == "GRU")
            .expect("GRU node");
        assert_eq!(gru_node.input.len(), 6);
        assert_eq!(gru_node.output.len(), 2);

        let hidden_size_attr = gru_node
            .attribute
            .iter()
            .find(|a| a.name == "hidden_size")
            .expect("hidden_size attr");
        assert_eq!(hidden_size_attr.i, 4);

        let lbr_attr = gru_node
            .attribute
            .iter()
            .find(|a| a.name == "linear_before_reset")
            .expect("linear_before_reset attr");
        assert_eq!(lbr_attr.i, 0);

        let has_bias_reorder = graph_proto
            .node
            .iter()
            .any(|n| n.name.contains("_b_reorder") && n.op_type == "Concat");
        assert!(has_bias_reorder);

        let has_output_squeeze = graph_proto
            .node
            .iter()
            .any(|n| n.name.contains("_squeeze_h") && n.op_type == "Squeeze");
        assert!(has_output_squeeze);
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn test_reshape_dynamic_new_shape_builds_runtime_shape_tensor() {
        let operands = vec![
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![
                        Dimension::Dynamic(DynamicDimension {
                            name: "batch".to_string(),
                            max_size: 8,
                        }),
                        Dimension::Static(2),
                        Dimension::Static(2),
                    ],
                    pending_permutation: vec![],
                },
                name: Some("input".to_string()),
            },
            Operand {
                kind: OperandKind::Output,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![
                        Dimension::Dynamic(DynamicDimension {
                            name: "batch".to_string(),
                            max_size: 8,
                        }),
                        Dimension::Static(4),
                    ],
                    pending_permutation: vec![],
                },
                name: Some("output".to_string()),
            },
        ];

        let operations = vec![Operation {
            op_type: "reshape".to_string(),
            input_operands: vec![0],
            output_operand: Some(1),
            output_operands: vec![],
            attributes: serde_json::json!({
                "newShape": [
                    { "name": "batch", "maxSize": 8 },
                    4
                ]
            }),
            label: None,
        }];

        let graph = GraphInfo {
            operands,
            input_operands: vec![0],
            output_operands: vec![1],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let model =
            ModelProto::decode(OnnxConverter.convert(&graph).unwrap().data.as_slice()).unwrap();
        let gp = model.graph.unwrap();
        assert!(gp.node.iter().any(|n| n.op_type == "Reshape"));
        assert!(gp.node.iter().any(|n| n.op_type == "Shape"));
        assert!(gp.node.iter().any(|n| n.op_type == "Gather"));
        assert!(gp.node.iter().any(|n| n.op_type == "Concat"));
    }

    #[cfg(feature = "dynamic-inputs")]
    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn test_expand_dynamic_new_shape_builds_runtime_shape_tensor() {
        let operands = vec![
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![
                        Dimension::Dynamic(DynamicDimension {
                            name: "batch".to_string(),
                            max_size: 8,
                        }),
                        Dimension::Static(1),
                    ],
                    pending_permutation: vec![],
                },
                name: Some("input".to_string()),
            },
            Operand {
                kind: OperandKind::Output,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![
                        Dimension::Dynamic(DynamicDimension {
                            name: "batch".to_string(),
                            max_size: 8,
                        }),
                        Dimension::Static(4),
                    ],
                    pending_permutation: vec![],
                },
                name: Some("output".to_string()),
            },
        ];

        let operations = vec![Operation {
            op_type: "expand".to_string(),
            input_operands: vec![0],
            output_operand: Some(1),
            output_operands: vec![],
            attributes: serde_json::json!({
                "newShape": [
                    { "name": "batch", "maxSize": 8 },
                    4
                ]
            }),
            label: None,
        }];

        let graph = GraphInfo {
            operands,
            input_operands: vec![0],
            output_operands: vec![1],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let model =
            ModelProto::decode(OnnxConverter.convert(&graph).unwrap().data.as_slice()).unwrap();
        let gp = model.graph.unwrap();
        assert!(gp.node.iter().any(|n| n.op_type == "Expand"));
        assert!(gp.node.iter().any(|n| n.op_type == "Shape"));
        assert!(gp.node.iter().any(|n| n.op_type == "Gather"));
        assert!(gp.node.iter().any(|n| n.op_type == "Concat"));
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn test_batchnorm_dynamic_channel_builds_runtime_default_scale_bias() {
        let operands = vec![
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![
                        Dimension::Static(1),
                        Dimension::Dynamic(DynamicDimension {
                            name: "channels".to_string(),
                            max_size: 8,
                        }),
                        Dimension::Static(4),
                        Dimension::Static(4),
                    ],
                    pending_permutation: vec![],
                },
                name: Some("input".to_string()),
            },
            Operand {
                kind: OperandKind::Constant,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: s(&[8]),
                    pending_permutation: vec![],
                },
                name: Some("mean".to_string()),
            },
            Operand {
                kind: OperandKind::Constant,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: s(&[8]),
                    pending_permutation: vec![],
                },
                name: Some("variance".to_string()),
            },
            Operand {
                kind: OperandKind::Output,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![
                        Dimension::Static(1),
                        Dimension::Dynamic(DynamicDimension {
                            name: "channels".to_string(),
                            max_size: 8,
                        }),
                        Dimension::Static(4),
                        Dimension::Static(4),
                    ],
                    pending_permutation: vec![],
                },
                name: Some("output".to_string()),
            },
        ];

        let operations = vec![Operation {
            op_type: "batchNormalization".to_string(),
            input_operands: vec![0, 1, 2], // no scale/bias operands, defaults required
            output_operand: Some(3),
            output_operands: vec![],
            attributes: serde_json::json!({
                "axis": 1,
                "hasScale": false,
                "hasBias": false,
                "epsilon": 1e-5
            }),
            label: None,
        }];

        let graph = GraphInfo {
            operands,
            input_operands: vec![0],
            output_operands: vec![3],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let model =
            ModelProto::decode(OnnxConverter.convert(&graph).unwrap().data.as_slice()).unwrap();
        let gp = model.graph.unwrap();

        assert!(gp.node.iter().any(|n| n.op_type == "Shape"));
        assert!(gp.node.iter().any(|n| n.op_type == "Gather"));
        assert!(gp.node.iter().filter(|n| n.op_type == "Expand").count() >= 2);
        assert!(gp.node.iter().any(|n| n.op_type == "BatchNormalization"));
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn test_layernorm_dynamic_axes_builds_runtime_default_scale_bias() {
        let operands = vec![
            Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![
                        Dimension::Static(2),
                        Dimension::Dynamic(DynamicDimension {
                            name: "seq".to_string(),
                            max_size: 16,
                        }),
                        Dimension::Static(32),
                    ],
                    pending_permutation: vec![],
                },
                name: Some("input".to_string()),
            },
            Operand {
                kind: OperandKind::Output,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![
                        Dimension::Static(2),
                        Dimension::Dynamic(DynamicDimension {
                            name: "seq".to_string(),
                            max_size: 16,
                        }),
                        Dimension::Static(32),
                    ],
                    pending_permutation: vec![],
                },
                name: Some("output".to_string()),
            },
        ];

        let operations = vec![Operation {
            op_type: "layerNormalization".to_string(),
            input_operands: vec![0], // no explicit scale/bias operands
            output_operand: Some(1),
            output_operands: vec![],
            attributes: serde_json::json!({
                "axes": [1],
                "hasScale": false,
                "hasBias": true,
                "epsilon": 1e-5
            }),
            label: None,
        }];

        let graph = GraphInfo {
            operands,
            input_operands: vec![0],
            output_operands: vec![1],
            operations,
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let model =
            ModelProto::decode(OnnxConverter.convert(&graph).unwrap().data.as_slice()).unwrap();
        let gp = model.graph.unwrap();

        assert!(gp.node.iter().any(|n| n.op_type == "Shape"));
        assert!(gp.node.iter().any(|n| n.op_type == "Slice"));
        assert!(gp.node.iter().filter(|n| n.op_type == "Expand").count() >= 2);
        assert!(gp.node.iter().any(|n| n.op_type == "LayerNormalization"));
    }
}

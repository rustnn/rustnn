use crate::converters::ConvertedGraph;
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo};
use crate::protos::coreml::specification::{
    AcosLayerParams, AcoshLayerParams, ActivationParams, ActivationReLu, ActivationSigmoid,
    AddLayerParams, ArrayFeatureType, AsinLayerParams, AsinhLayerParams, AtanLayerParams,
    AtanhLayerParams, CeilLayerParams, ConvolutionLayerParams, CosLayerParams, CoshLayerParams,
    EqualLayerParams, ErfLayerParams, FeatureDescription, FeatureType, FloorLayerParams,
    GreaterEqualLayerParams, GreaterThanLayerParams, InnerProductLayerParams, LessEqualLayerParams,
    LessThanLayerParams, LoadConstantLayerParams, LogicalAndLayerParams, LogicalNotLayerParams,
    LogicalOrLayerParams, LogicalXorLayerParams, Model, ModelDescription, MultiplyLayerParams,
    NeuralNetwork, NeuralNetworkLayer, PoolingLayerParams, ReduceL1LayerParams,
    ReduceL2LayerParams, ReduceLogSumExpLayerParams, ReduceLogSumLayerParams, ReduceMaxLayerParams,
    ReduceMeanLayerParams, ReduceMinLayerParams, ReduceProdLayerParams, ReduceSumLayerParams,
    ReduceSumSquareLayerParams, RoundLayerParams, SignLayerParams, SinLayerParams, SinhLayerParams,
    SoftmaxLayerParams, TanLayerParams, TanhLayerParams, UnaryFunctionLayerParams, WeightParams,
    activation_params::NonlinearityType, array_feature_type::ArrayDataType, feature_type, model,
    neural_network_layer::Layer, pooling_layer_params, unary_function_layer_params,
};
use prost::Message;
use prost::bytes::Bytes;

#[derive(Default)]
pub struct CoremlConverter;

impl CoremlConverter {
    fn operand_name(graph: &GraphInfo, id: u32) -> String {
        graph
            .operand(id)
            .and_then(|op| op.name.clone())
            .unwrap_or_else(|| format!("operand_{}", id))
    }

    fn coerce_shape(shape: &[u32]) -> Vec<i64> {
        let mut dims: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
        match dims.len() {
            0 => vec![1],
            1 => dims,
            2 => {
                let mut with_batch = vec![1];
                with_batch.append(&mut dims);
                with_batch
            }
            3 => dims,
            _ => {
                let prod: i64 = dims.iter().product();
                vec![prod]
            }
        }
    }

    fn feature_type(desc: &crate::graph::OperandDescriptor) -> FeatureType {
        let shape = Self::coerce_shape(&desc.shape);
        FeatureType {
            r#type: Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                shape,
                data_type: match desc.data_type {
                    DataType::Float32 => ArrayDataType::Float32 as i32,
                    DataType::Float16 => ArrayDataType::Float16 as i32,
                    _ => ArrayDataType::Int32 as i32,
                },
                ..Default::default()
            })),
            ..Default::default()
        }
    }
}

impl crate::converters::GraphConverter for CoremlConverter {
    fn format(&self) -> &'static str {
        "coreml"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        let inputs = graph
            .input_operands
            .iter()
            .map(|id| {
                let operand = graph
                    .operand(*id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
                Ok(FeatureDescription {
                    name: Self::operand_name(graph, *id),
                    r#type: Some(Self::feature_type(&operand.descriptor)),
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let outputs = graph
            .output_operands
            .iter()
            .map(|id| {
                let operand = graph
                    .operand(*id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
                Ok(FeatureDescription {
                    name: Self::operand_name(graph, *id),
                    r#type: Some(Self::feature_type(&operand.descriptor)),
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let mut layers = Vec::new();

        // Emit constants as LoadConstant layers
        for (id, data) in &graph.constant_operand_ids_to_handles {
            let operand = graph
                .operand(*id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
            let name = Self::operand_name(graph, *id);
            let shape: Vec<u64> = Self::coerce_shape(&operand.descriptor.shape)
                .into_iter()
                .map(|d| d as u64)
                .collect();
            let float_value = if matches!(operand.descriptor.data_type, DataType::Float32) {
                bytemuck::cast_slice::<u8, f32>(&data.data).to_vec()
            } else {
                Vec::new()
            };

            let weight = WeightParams {
                float_value,
                raw_value: if matches!(operand.descriptor.data_type, DataType::Float32) {
                    Bytes::new()
                } else {
                    Bytes::from(data.data.clone())
                },
                ..Default::default()
            };
            layers.push(NeuralNetworkLayer {
                name: format!("const_{}", name),
                input: vec![],
                output: vec![name],
                layer: Some(Layer::LoadConstant(LoadConstantLayerParams {
                    shape: shape.clone(),
                    data: Some(weight.clone()),
                    ..Default::default()
                })),
                ..Default::default()
            });
        }

        for (idx, op) in graph.operations.iter().enumerate() {
            let layer_name = op
                .label
                .as_ref()
                .cloned()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("{}_{}", op.op_type, idx));
            let input_names = op
                .input_operands
                .iter()
                .map(|id| Self::operand_name(graph, *id))
                .collect();
            let output_names = vec![Self::operand_name(graph, op.output_operand)];

            let layer = if op.op_type.eq_ignore_ascii_case("add") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Add(AddLayerParams {
                        alpha: 0.0,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("matmul") {
                let rhs_id =
                    *op.input_operands
                        .get(1)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml".to_string(),
                            reason: "matmul requires two inputs".to_string(),
                        })?;
                let rhs_operand = graph
                    .operand(rhs_id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: rhs_id })?;
                let rhs_shape = rhs_operand.descriptor.shape.clone();
                if rhs_shape.len() != 2 {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml".to_string(),
                        reason: format!("matmul constant must be 2D, got shape {:?}", rhs_shape),
                    });
                }
                let (in_ch, out_ch) = (rhs_shape[0] as usize, rhs_shape[1] as usize);
                let rhs_data = graph
                    .constant_operand_ids_to_handles
                    .get(&rhs_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml".to_string(),
                        reason: "matmul weights must be constant".to_string(),
                    })?;
                let floats: &[f32] = bytemuck::try_cast_slice(&rhs_data.data).map_err(|_| {
                    GraphError::ConversionFailed {
                        format: "coreml".to_string(),
                        reason: "matmul weights must be float32".to_string(),
                    }
                })?;
                if floats.len() != in_ch * out_ch {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml".to_string(),
                        reason: format!(
                            "matmul weight size mismatch (expected {} got {})",
                            in_ch * out_ch,
                            floats.len()
                        ),
                    });
                }
                let mut transposed = Vec::with_capacity(floats.len());
                for o in 0..out_ch {
                    for i in 0..in_ch {
                        transposed.push(floats[i * out_ch + o]);
                    }
                }
                let weight = WeightParams {
                    float_value: transposed,
                    raw_value: Bytes::new(),
                    ..Default::default()
                };
                NeuralNetworkLayer {
                    name: layer_name,
                    input: vec![input_names.get(0).cloned().ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml".to_string(),
                            reason: "matmul missing lhs input".to_string(),
                        }
                    })?],
                    output: output_names,
                    layer: Some(Layer::InnerProduct(InnerProductLayerParams {
                        input_channels: in_ch as u64,
                        output_channels: out_ch as u64,
                        has_bias: false,
                        weights: Some(weight),
                        bias: None,
                        int8_dynamic_quantize: false,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("relu") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Activation(ActivationParams {
                        nonlinearity_type: Some(NonlinearityType::ReLu(ActivationReLu {})),
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("sigmoid") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Activation(ActivationParams {
                        nonlinearity_type: Some(NonlinearityType::Sigmoid(ActivationSigmoid {})),
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("tanh") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Tanh(TanhLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("softmax") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Softmax(SoftmaxLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("conv2d") {
                // Get filter operand (second input)
                let filter_id =
                    *op.input_operands
                        .get(1)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml".to_string(),
                            reason: "conv2d requires filter input".to_string(),
                        })?;

                let filter_operand = graph
                    .operand(filter_id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: filter_id })?;
                let filter_shape = filter_operand.descriptor.shape.clone();

                // Filter shape: [out_channels, in_channels, height, width] (OIHW layout)
                if filter_shape.len() != 4 {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml".to_string(),
                        reason: format!("conv2d filter must be 4D, got shape {:?}", filter_shape),
                    });
                }

                let (out_channels, kernel_channels, kernel_h, kernel_w) = (
                    filter_shape[0],
                    filter_shape[1],
                    filter_shape[2],
                    filter_shape[3],
                );

                // Get filter weights from constant data
                let filter_data = graph
                    .constant_operand_ids_to_handles
                    .get(&filter_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml".to_string(),
                        reason: "conv2d filter must be constant".to_string(),
                    })?;

                let floats: &[f32] = bytemuck::try_cast_slice(&filter_data.data).map_err(|_| {
                    GraphError::ConversionFailed {
                        format: "coreml".to_string(),
                        reason: "conv2d filter must be float32".to_string(),
                    }
                })?;

                // Parse attributes
                let strides = op
                    .attributes
                    .get("strides")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect::<Vec<u64>>())
                    .unwrap_or_else(|| vec![1, 1]);

                let dilations = op
                    .attributes
                    .get("dilations")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect::<Vec<u64>>())
                    .unwrap_or_else(|| vec![1, 1]);

                let groups = op
                    .attributes
                    .get("groups")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1);

                let weight = WeightParams {
                    float_value: floats.to_vec(),
                    raw_value: Bytes::new(),
                    ..Default::default()
                };

                NeuralNetworkLayer {
                    name: layer_name,
                    input: vec![input_names.get(0).cloned().ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml".to_string(),
                            reason: "conv2d missing input".to_string(),
                        }
                    })?],
                    output: output_names,
                    layer: Some(Layer::Convolution(ConvolutionLayerParams {
                        output_channels: out_channels as u64,
                        kernel_channels: kernel_channels as u64,
                        n_groups: groups,
                        kernel_size: vec![kernel_h as u64, kernel_w as u64],
                        stride: strides,
                        dilation_factor: dilations,
                        is_deconvolution: false,
                        has_bias: false,
                        weights: Some(weight),
                        bias: None,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("convTranspose2d") {
                // Get filter operand (second input)
                let filter_id =
                    *op.input_operands
                        .get(1)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml".to_string(),
                            reason: "convTranspose2d requires filter input".to_string(),
                        })?;

                let filter_operand = graph
                    .operand(filter_id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: filter_id })?;
                let filter_shape = filter_operand.descriptor.shape.clone();

                // Filter shape: [in_channels, out_channels/groups, height, width] (transpose conv layout)
                if filter_shape.len() != 4 {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml".to_string(),
                        reason: format!(
                            "convTranspose2d filter must be 4D, got shape {:?}",
                            filter_shape
                        ),
                    });
                }

                let (in_channels, out_channels_per_group, kernel_h, kernel_w) = (
                    filter_shape[0],
                    filter_shape[1],
                    filter_shape[2],
                    filter_shape[3],
                );

                // Get filter weights from constant data
                let filter_data = graph
                    .constant_operand_ids_to_handles
                    .get(&filter_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml".to_string(),
                        reason: "convTranspose2d filter must be constant".to_string(),
                    })?;

                let floats: &[f32] = bytemuck::try_cast_slice(&filter_data.data).map_err(|_| {
                    GraphError::ConversionFailed {
                        format: "coreml".to_string(),
                        reason: "convTranspose2d filter must be float32".to_string(),
                    }
                })?;

                // Parse attributes
                let strides = op
                    .attributes
                    .get("strides")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect::<Vec<u64>>())
                    .unwrap_or_else(|| vec![1, 1]);

                let dilations = op
                    .attributes
                    .get("dilations")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect::<Vec<u64>>())
                    .unwrap_or_else(|| vec![1, 1]);

                let groups = op
                    .attributes
                    .get("groups")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1);

                let out_channels = out_channels_per_group * (groups as u32);

                let weight = WeightParams {
                    float_value: floats.to_vec(),
                    raw_value: Bytes::new(),
                    ..Default::default()
                };

                NeuralNetworkLayer {
                    name: layer_name,
                    input: vec![input_names.get(0).cloned().ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml".to_string(),
                            reason: "convTranspose2d missing input".to_string(),
                        }
                    })?],
                    output: output_names,
                    layer: Some(Layer::Convolution(ConvolutionLayerParams {
                        output_channels: out_channels as u64,
                        kernel_channels: in_channels as u64,
                        n_groups: groups,
                        kernel_size: vec![kernel_h as u64, kernel_w as u64],
                        stride: strides,
                        dilation_factor: dilations,
                        is_deconvolution: true, // Key difference: this is a transposed conv
                        has_bias: false,
                        weights: Some(weight),
                        bias: None,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("averagePool2d")
                || op.op_type.eq_ignore_ascii_case("maxPool2d")
            {
                // Get pool parameters from attributes
                let window_dimensions = op
                    .attributes
                    .get("windowDimensions")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|u| u as u64))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_else(|| vec![1, 1]);

                let strides = op
                    .attributes
                    .get("strides")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|u| u as u64))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_else(|| vec![1, 1]);

                let pads = op
                    .attributes
                    .get("pads")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|u| u as u64))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_else(|| vec![0, 0, 0, 0]);

                // CoreML padding format: [top, bottom, left, right]
                let padding_top = pads[0];
                let padding_bottom = pads[2];
                let padding_left = pads[1];
                let padding_right = pads[3];

                // Determine pooling type
                let pooling_type = if op.op_type.eq_ignore_ascii_case("averagePool2d") {
                    pooling_layer_params::PoolingType::Average
                } else {
                    pooling_layer_params::PoolingType::Max
                };

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Pooling(PoolingLayerParams {
                        r#type: pooling_type as i32,
                        kernel_size: vec![window_dimensions[0], window_dimensions[1]],
                        stride: vec![strides[0], strides[1]],
                        avg_pool_exclude_padding: false,
                        global_pooling: false,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("globalAveragePool")
                || op.op_type.eq_ignore_ascii_case("globalMaxPool")
            {
                // Determine pooling type
                let pooling_type = if op.op_type.eq_ignore_ascii_case("globalAveragePool") {
                    pooling_layer_params::PoolingType::Average
                } else {
                    pooling_layer_params::PoolingType::Max
                };

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Pooling(PoolingLayerParams {
                        r#type: pooling_type as i32,
                        kernel_size: vec![], // Empty for global pooling
                        stride: vec![],      // Empty for global pooling
                        avg_pool_exclude_padding: false,
                        global_pooling: true, // Key: this makes it global pooling
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reduceSum") {
                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reduce_all = axes.is_empty();

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::ReduceSum(ReduceSumLayerParams {
                        axes,
                        keep_dims,
                        reduce_all,
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reduceMean") {
                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reduce_all = axes.is_empty();

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::ReduceMean(ReduceMeanLayerParams {
                        axes,
                        keep_dims,
                        reduce_all,
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reduceMax") {
                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reduce_all = axes.is_empty();

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::ReduceMax(ReduceMaxLayerParams {
                        axes,
                        keep_dims,
                        reduce_all,
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reduceMin") {
                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reduce_all = axes.is_empty();

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::ReduceMin(ReduceMinLayerParams {
                        axes,
                        keep_dims,
                        reduce_all,
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reduceProduct") {
                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reduce_all = axes.is_empty();

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::ReduceProd(ReduceProdLayerParams {
                        axes,
                        keep_dims,
                        reduce_all,
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reduceL1") {
                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reduce_all = axes.is_empty();

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::ReduceL1(ReduceL1LayerParams {
                        axes,
                        keep_dims,
                        reduce_all,
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reduceL2") {
                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reduce_all = axes.is_empty();

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::ReduceL2(ReduceL2LayerParams {
                        axes,
                        keep_dims,
                        reduce_all,
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reduceLogSum") {
                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reduce_all = axes.is_empty();

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::ReduceLogSum(ReduceLogSumLayerParams {
                        axes,
                        keep_dims,
                        reduce_all,
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reduceLogSumExp") {
                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reduce_all = axes.is_empty();

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::ReduceLogSumExp(ReduceLogSumExpLayerParams {
                        axes,
                        keep_dims,
                        reduce_all,
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reduceSumSquare") {
                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let reduce_all = axes.is_empty();

                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::ReduceSumSquare(ReduceSumSquareLayerParams {
                        axes,
                        keep_dims,
                        reduce_all,
                    })),
                    ..Default::default()
                }
            // Element-wise operations - Basic math
            } else if op.op_type.eq_ignore_ascii_case("abs") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Unary(UnaryFunctionLayerParams {
                        r#type: unary_function_layer_params::Operation::Abs as i32,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("ceil") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Ceil(CeilLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("floor") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Floor(FloorLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("round") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Round(RoundLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("neg") {
                // CoreML doesn't have a dedicated neg layer, use multiply by -1
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Multiply(MultiplyLayerParams { alpha: -1.0 })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("sign") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Sign(SignLayerParams {})),
                    ..Default::default()
                }
            // Element-wise operations - Exponential and logarithmic
            } else if op.op_type.eq_ignore_ascii_case("exp") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Unary(UnaryFunctionLayerParams {
                        r#type: unary_function_layer_params::Operation::Exp as i32,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("log") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Unary(UnaryFunctionLayerParams {
                        r#type: unary_function_layer_params::Operation::Log as i32,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("sqrt") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Unary(UnaryFunctionLayerParams {
                        r#type: unary_function_layer_params::Operation::Sqrt as i32,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("reciprocal") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Unary(UnaryFunctionLayerParams {
                        r#type: unary_function_layer_params::Operation::Inverse as i32,
                        ..Default::default()
                    })),
                    ..Default::default()
                }
            // Element-wise operations - Trigonometric
            } else if op.op_type.eq_ignore_ascii_case("sin") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Sin(SinLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("cos") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Cos(CosLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("tan") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Tan(TanLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("asin") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Asin(AsinLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("acos") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Acos(AcosLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("atan") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Atan(AtanLayerParams {})),
                    ..Default::default()
                }
            // Element-wise operations - Hyperbolic
            } else if op.op_type.eq_ignore_ascii_case("sinh") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Sinh(SinhLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("cosh") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Cosh(CoshLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("asinh") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Asinh(AsinhLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("acosh") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Acosh(AcoshLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("atanh") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Atanh(AtanhLayerParams {})),
                    ..Default::default()
                }
            // Element-wise operations - Special functions
            } else if op.op_type.eq_ignore_ascii_case("erf") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Erf(ErfLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("identity") {
                // CoreML doesn't have an identity layer, use multiply by 1
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Multiply(MultiplyLayerParams { alpha: 1.0 })),
                    ..Default::default()
                }
            // Logic operations - Comparison
            } else if op.op_type.eq_ignore_ascii_case("equal") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::Equal(EqualLayerParams { alpha: 0.0 })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("greater") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::GreaterThan(GreaterThanLayerParams { alpha: 0.0 })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("greaterOrEqual") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::GreaterEqual(GreaterEqualLayerParams { alpha: 0.0 })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("lesser") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::LessThan(LessThanLayerParams { alpha: 0.0 })),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("lesserOrEqual") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::LessEqual(LessEqualLayerParams { alpha: 0.0 })),
                    ..Default::default()
                }
            // Logic operations - Logical
            } else if op.op_type.eq_ignore_ascii_case("logicalNot") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::LogicalNot(LogicalNotLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("logicalAnd") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::LogicalAnd(LogicalAndLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("logicalOr") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::LogicalOr(LogicalOrLayerParams {})),
                    ..Default::default()
                }
            } else if op.op_type.eq_ignore_ascii_case("logicalXor") {
                NeuralNetworkLayer {
                    name: layer_name,
                    input: input_names,
                    output: output_names,
                    layer: Some(Layer::LogicalXor(LogicalXorLayerParams {})),
                    ..Default::default()
                }
            } else {
                return Err(GraphError::ConversionFailed {
                    format: "coreml".to_string(),
                    reason: format!("Unsupported op_type {}", op.op_type),
                });
            };
            layers.push(layer);
        }

        let nn = NeuralNetwork {
            layers,
            ..Default::default()
        };

        let description = ModelDescription {
            input: inputs,
            output: outputs,
            ..Default::default()
        };

        let model = Model {
            specification_version: 7,
            description: Some(description),
            r#type: Some(model::Type::NeuralNetwork(nn)),
            ..Default::default()
        };

        let data = model.encode_to_vec();

        Ok(ConvertedGraph {
            format: "coreml",
            content_type: "application/x-apple-mlmodel",
            data,
        })
    }
}

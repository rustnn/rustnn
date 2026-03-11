use crate::converters::operand_name;
use crate::graph::Dimension;
use crate::operator_enums::MLOperandDataType;
use crate::operator_options::{
    MLArgMinMaxOptions, MLBatchNormalizationOptions, MLClampOptions, MLConv2dOptions,
    MLConvTranspose2dOptions, MLCumulativeSumOptions, MLDimension, MLEluOptions, MLGatherOptions,
    MLGemmOptions, MLGruCellOptions, MLGruOptions, MLHardSigmoidOptions,
    MLInstanceNormalizationOptions, MLLayerNormalizationOptions, MLLeakyReluOptions,
    MLLinearOptions, MLPadOptions, MLPool2dOptions, MLReduceOptions, MLResample2dOptions,
    MLReverseOptions, MLScatterOptions, MLSliceOptions, MLSplitOptions, MLTransposeOptions,
    MLTriangularOptions,
};
use crate::{DataType, GraphError};
use js_sys::{Reflect, Uint8Array};
use pollster::FutureExt as _;
use wasm_bindgen::JsValue;
// for future.block_on, should transform to async
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    MlArgMinMaxOptions, MlBatchNormalizationOptions, MlClampOptions, MlContext, MlContextOptions,
    MlConv2dFilterOperandLayout, MlConv2dOptions, MlConvTranspose2dFilterOperandLayout,
    MlConvTranspose2dOptions, MlCumulativeSumOptions, MlEluOptions, MlGatherOptions, MlGemmOptions,
    MlGraphBuilder, MlGruCellOptions, MlGruOptions, MlGruWeightLayout, MlHardSigmoidOptions,
    MlInputOperandLayout, MlInstanceNormalizationOptions, MlInterpolationMode,
    MlLayerNormalizationOptions, MlLeakyReluOptions, MlLinearOptions, MlOperand, MlOperandDataType,
    MlOperandDescriptor, MlOperatorOptions, MlPadOptions, MlPaddingMode, MlPool2dOptions,
    MlPowerPreference, MlRecurrentNetworkActivation, MlRecurrentNetworkDirection, MlReduceOptions,
    MlResample2dOptions, MlReverseOptions, MlRoundingType, MlScatterOptions, MlSliceOptions,
    MlSplitOptions, MlTransposeOptions, MlTriangularOptions, window,
};

fn u32_slice_to_js(v: &[u32]) -> Vec<js_sys::Number> {
    v.iter().map(|&x| js_sys::Number::from(x as f64)).collect()
}
fn f32_slice_to_js(v: &[f32]) -> Vec<js_sys::Number> {
    v.iter().map(|&x| js_sys::Number::from(x as f64)).collect()
}

type Result<T> = std::result::Result<T, GraphError>;

pub struct WebNNConverter {
    context: js_sys::Promise<MlContext>,
}

impl Default for WebNNConverter {
    fn default() -> Self {
        let window = window().expect("no global `window` exists");
        let navigator = window.navigator();
        let ml = navigator.ml();

        let options = MlContextOptions::new();
        options.set_accelerated(true);
        options.set_power_preference(MlPowerPreference::HighPerformance);
        let promise = ml.create_context_with_ml_context_options(&options);
        Self { context: promise }
    }
}
use super::ConvertedGraph;
use crate::GraphInfo;

enum WebNNConverterError {
    InvalidId(u32),
    IdNotAConstant(u32),
    DynamicShapeNotSupported,
    UnsupportedDataType(DataType),
    FailedToCreateMlContext,
    FailedToCreateMlBuilder,
    FailedToBuildGraph(JsValue),
    FailedRepresentAsF64(serde_json::Number),
}

impl From<WebNNConverterError> for GraphError {
    fn from(error: WebNNConverterError) -> Self {
        GraphError::ConversionFailed {
            format: "webnn".to_string(),
            reason: match error {
                WebNNConverterError::InvalidId(id) => format!("Invalid id {id} in graph"),
                WebNNConverterError::UnsupportedDataType(data_type) => {
                    format!("Unsupported data type {data_type:?} in graph")
                }
                WebNNConverterError::FailedToCreateMlContext => "Failed to create MlContext".into(),
                WebNNConverterError::FailedToCreateMlBuilder => "Failed to create MlBuilder".into(),
                WebNNConverterError::IdNotAConstant(id) => {
                    format!("Operand {id} is not a constant")
                }
                WebNNConverterError::DynamicShapeNotSupported => {
                    "The WebNN backend does not support dynamic shapes (e.g. the Shape operator)"
                        .to_string()
                }
                WebNNConverterError::FailedToBuildGraph(err) => {
                    format!("Failed to build graph using browser's MlGraphBuilder.build: {err:?}")
                }
                WebNNConverterError::FailedRepresentAsF64(number) => {
                    format!("Failed to respresent as f64: {number:?}")
                }
            },
        }
    }
}

fn get_constant_data(graph: &GraphInfo, operand_id: u32) -> Result<&[u8]> {
    graph
        .constant_operand_ids_to_handles
        .get(&operand_id)
        .map(|constant_data| constant_data.data.as_slice())
        .ok_or(WebNNConverterError::IdNotAConstant(operand_id).into())
}

fn get_operand(operands: &[Option<MlOperand>], index: u32) -> &MlOperand {
    operands[index as usize]
        .as_ref()
        .expect("operand not found")
}

impl WebNNConverter {
    pub async fn convert_async(
        &self,
        context: &MlContext,
        graph_info: &GraphInfo,
    ) -> Result<ConvertedGraph> {
        let builder = MlGraphBuilder::new(context)
            .map_err(|_e| WebNNConverterError::FailedToCreateMlBuilder)?;

        let mut operands = graph_info
            .operands
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let dtype = dtype_to_dtype(o.descriptor.data_type)?;
                let shape = shape_to_shape(&o.descriptor.shape);
                let desc = MlOperandDescriptor::new(dtype, &shape);

                Ok(match o.kind {
                    crate::OperandKind::Input => {
                        Some(builder.input(o.name.as_deref().unwrap_or(""), &desc))
                    }
                    crate::OperandKind::Constant => {
                        Some(builder.constant_with_ml_operand_descriptor_and_u8_array(
                            &desc,
                            &Uint8Array::new_from_slice(get_constant_data(graph_info, i as u32)?),
                        ))
                    }
                    crate::OperandKind::Output => None,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        for op in graph_info.operations.iter() {
            for id in op.input_operands().iter().copied() {
                let invalid = operands
                    .get(id as usize)
                    .map(|op: &Option<MlOperand>| op.is_none())
                    .unwrap_or(false);
                if invalid {
                    return Err(WebNNConverterError::InvalidId(id).into());
                }
            }
            let ml_opts = MlOperatorOptions::new();
            ml_opts.set_label(op.attributes().label());
            let out = match op {
                crate::Operation::Add {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.add_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Sub {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.sub_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Mul {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.mul_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Div {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.div_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Pow {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.pow_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Max {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.max_with_options(
                        get_operand(&operands, *a),
                        get_operand(&operands, *b),
                        &ml_opts,
                    )]
                }
                crate::Operation::Min {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.min_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Matmul {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.matmul_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Equal {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.equal_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Greater {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.greater_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::GreaterOrEqual {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.greater_or_equal_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Lesser {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.lesser_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::LesserOrEqual {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.lesser_or_equal_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::NotEqual {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.not_equal_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Abs {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.abs_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Ceil {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.ceil_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Cos {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.cos_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Exp {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.exp_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Floor {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.floor_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Log {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.log_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Neg {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.neg_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Relu {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.relu_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Sigmoid {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.sigmoid_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Sin {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.sin_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Sqrt {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.sqrt_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Tan {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.tan_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Tanh {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.tanh_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Erf {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.erf_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::Reciprocal {
                    input,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.reciprocal_with_options(get_operand(&operands, *input), &ml_opts)]
                }
                crate::Operation::Sign {
                    input,
                    options: _,
                    outputs: _,
                } => vec![builder.sign_with_options(get_operand(&operands, *input), &ml_opts)],
                crate::Operation::LogicalAnd {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.logical_and_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::LogicalOr {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.logical_or_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::LogicalNot {
                    input,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.logical_not_with_options(get_operand(&operands, *input), &ml_opts)]
                }
                crate::Operation::LogicalXor {
                    a,
                    b,
                    options: _,
                    outputs: _,
                } => vec![builder.logical_xor_with_options(
                    get_operand(&operands, *a),
                    get_operand(&operands, *b),
                    &ml_opts,
                )],
                crate::Operation::Where {
                    condition,
                    true_value,
                    false_value,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.where_with_options(
                        get_operand(&operands, *condition),
                        get_operand(&operands, *true_value),
                        get_operand(&operands, *false_value),
                        &ml_opts,
                    )]
                }
                crate::Operation::Identity {
                    input,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.identity_with_options(get_operand(&operands, *input), &ml_opts)]
                }
                crate::Operation::ArgMin {
                    input,
                    axis,
                    options,
                    outputs: _,
                } => vec![builder.arg_min_with_options(
                    get_operand(&operands, *input),
                    *axis,
                    &to_min_max_options(options)?,
                )],
                crate::Operation::ArgMax {
                    input,
                    axis,
                    options,
                    outputs: _,
                } => vec![builder.arg_max_with_options(
                    get_operand(&operands, *input),
                    *axis,
                    &to_min_max_options(options)?,
                )],
                crate::Operation::BatchNormalization {
                    input,
                    mean,
                    variance,
                    options,
                    outputs: _,
                } => vec![builder.batch_normalization_with_options(
                    get_operand(&operands, *input),
                    get_operand(&operands, *mean),
                    get_operand(&operands, *variance),
                    &to_batch_norm_options(options, &operands)?,
                )],
                crate::Operation::Cast {
                    input,
                    data_type,
                    options: _,
                    outputs: _,
                } => vec![builder.cast_with_options(
                    get_operand(&operands, *input),
                    operand_dtype_to_dtype(*data_type)?,
                    &ml_opts,
                )],
                crate::Operation::Clamp {
                    input,
                    options,
                    outputs: _,
                } => vec![builder.clamp_with_options(
                    get_operand(&operands, *input),
                    &to_clamp_options(options)?,
                )],
                crate::Operation::Constant {
                    options: _,
                    outputs: _,
                } => todo!(),
                crate::Operation::Conv2d {
                    input,
                    filter,
                    options,
                    outputs: _,
                } => vec![builder.conv2d_with_options(
                    get_operand(&operands, *input),
                    get_operand(&operands, *filter),
                    &to_conv2d_options(options, &operands)?,
                )],
                crate::Operation::ConvTranspose2d {
                    input,
                    filter,
                    options,
                    outputs: _,
                } => vec![builder.conv_transpose2d_with_options(
                    get_operand(&operands, *input),
                    get_operand(&operands, *filter),
                    &to_conv_transpose2d_options(options, &operands)?,
                )],
                crate::Operation::Concat {
                    inputs,
                    axis,
                    options: _,
                    outputs: _,
                } => vec![
                    builder.concat_with_options(
                        inputs
                            .iter()
                            .map(|i| get_operand(&operands, *i).clone())
                            .collect::<Vec<_>>()
                            .as_slice(),
                        *axis,
                        &ml_opts,
                    ),
                ],
                crate::Operation::CumulativeSum {
                    input,
                    axis,
                    options,
                    outputs: _,
                } => vec![builder.cumulative_sum_with_options(
                    get_operand(&operands, *input),
                    *axis,
                    &to_cumulative_sum_options(options)?,
                )],
                crate::Operation::Expand {
                    input,
                    new_shape,
                    options: _,
                    outputs: _,
                } => vec![builder.expand_with_options(
                    get_operand(&operands, *input),
                    &shape_to_mlshape(new_shape),
                    &ml_opts,
                )],
                crate::Operation::Elu {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.elu_with_options(
                        get_operand(&operands, *input),
                        &to_elu_options(options)?,
                    )]
                }
                crate::Operation::Gather {
                    input,
                    indices,
                    batch_dimensions: _,
                    options,
                    outputs: _,
                } => {
                    vec![builder.gather_with_options(
                        get_operand(&operands, *input),
                        get_operand(&operands, *indices),
                        &to_gather_options(options)?,
                    )]
                }
                crate::Operation::GatherElements {
                    input,
                    indices,
                    batch_dimensions: _,
                    options,
                    outputs: _,
                } => {
                    vec![builder.gather_elements_with_options(
                        get_operand(&operands, *input),
                        get_operand(&operands, *indices),
                        &to_gather_options(options)?,
                    )]
                }
                crate::Operation::Gemm {
                    a,
                    b,
                    options,
                    outputs: _,
                } => {
                    vec![builder.gemm_with_options(
                        get_operand(&operands, *a),
                        get_operand(&operands, *b),
                        &to_gemm_options(options)?,
                    )]
                }
                crate::Operation::Gru {
                    input,
                    weight,
                    recurrence,
                    steps,
                    hidden_size,
                    options,
                    outputs: _,
                } => builder
                    .gru_with_options(
                        get_operand(&operands, *input),
                        get_operand(&operands, *weight),
                        get_operand(&operands, *recurrence),
                        *steps,
                        *hidden_size,
                        &to_gru_options(options, &operands)?,
                    )
                    .to_vec(),
                crate::Operation::GruCell {
                    input,
                    weight,
                    recurrence,
                    hidden_state,
                    hidden_size,
                    options,
                    outputs: _,
                } => {
                    vec![builder.gru_cell_with_options(
                        get_operand(&operands, *input),
                        get_operand(&operands, *weight),
                        get_operand(&operands, *recurrence),
                        get_operand(&operands, *hidden_state),
                        *hidden_size,
                        &to_gru_cell_options(options, &operands)?,
                    )]
                }
                crate::Operation::HardSigmoid {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.hard_sigmoid_with_options(
                        get_operand(&operands, *input),
                        &to_hard_sigmoid_options(options)?,
                    )]
                }
                crate::Operation::HardSwish {
                    input,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.hard_swish_with_options(get_operand(&operands, *input), &ml_opts)]
                }
                crate::Operation::InstanceNormalization {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.instance_normalization_with_options(
                        get_operand(&operands, *input),
                        &to_instance_normalization_options(options, &operands)?,
                    )]
                }
                crate::Operation::LayerNormalization {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.layer_normalization_with_options(
                        get_operand(&operands, *input),
                        &to_layer_normalization_options(options, &operands)?,
                    )]
                }
                crate::Operation::LeakyRelu {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.leaky_relu_with_options(
                        get_operand(&operands, *input),
                        &to_leaky_relu_options(options)?,
                    )]
                }
                crate::Operation::Linear {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.linear_with_options(
                        get_operand(&operands, *input),
                        &to_linear_options(options)?,
                    )]
                }
                crate::Operation::Lstm {
                    input: _,
                    weight: _,
                    recurrence: _,
                    options: _,
                    outputs: _,
                } => {
                    todo!()
                    //vec![builder.lstm_with_options(
                    //get_operand(&operands, *input),
                    //get_operand(&operands, *input),
                    //get_operand(&operands, *recurrence),
                    //todo!(), // TODO
                    //todo!(), // TODO
                    //todo!(),
                    //)]
                }
                crate::Operation::LstmCell {
                    ..
                    //input,
                    //weight,
                    //recurrence,
                    //hidden_state,
                    //cell_state,
                    //options,
                    //outputs: _,
                } => {
                    todo!()
                }
                crate::Operation::Pad {
                    input,
                    beginning_padding,
                    ending_padding,
                    options,
                    outputs: _,
                } => {
                    vec![builder.pad_with_options(
                        get_operand(&operands, *input),
                        &u32_slice_to_js(beginning_padding),
                        &u32_slice_to_js(ending_padding),
                        &to_pad_options(options)?,
                    )]
                }
                crate::Operation::AveragePool2d {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.average_pool2d_with_options(
                        get_operand(&operands, *input),
                        &to_pool2d_options(options)?,
                    )]
                }
                crate::Operation::MaxPool2d {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.max_pool2d_with_options(
                        get_operand(&operands, *input),
                        &to_pool2d_options(options)?,
                    )]
                }
                crate::Operation::L2Pool2d {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.l2_pool2d_with_options(
                        get_operand(&operands, *input),
                        &to_pool2d_options(options)?,
                    )]
                }
                // halluzination?
                crate::Operation::GlobalAveragePool {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.average_pool2d_with_options(
                        get_operand(&operands, *input),
                        &to_pool2d_options(options)?,
                    )]
                }
                // halluzination?
                crate::Operation::GlobalMaxPool {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.max_pool2d_with_options(
                        get_operand(&operands, *input),
                        &to_pool2d_options(options)?,
                    )]
                }
                crate::Operation::ReduceSum {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reduce_sum_with_options(
                        get_operand(&operands, *input),
                        &to_reduce_options(options)?,
                    )]
                }
                crate::Operation::ReduceMean {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reduce_mean_with_options(
                        get_operand(&operands, *input),
                        &to_reduce_options(options)?,
                    )]
                }
                crate::Operation::ReduceMax {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reduce_max_with_options(
                        get_operand(&operands, *input),
                        &to_reduce_options(options)?,
                    )]
                }
                crate::Operation::ReduceMin {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reduce_min_with_options(
                        get_operand(&operands, *input),
                        &to_reduce_options(options)?,
                    )]
                }
                crate::Operation::ReduceProduct {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reduce_product_with_options(
                        get_operand(&operands, *input),
                        &to_reduce_options(options)?,
                    )]
                }
                crate::Operation::ReduceL1 {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reduce_l1_with_options(
                        get_operand(&operands, *input),
                        &to_reduce_options(options)?,
                    )]
                }
                crate::Operation::ReduceL2 {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reduce_l2_with_options(
                        get_operand(&operands, *input),
                        &to_reduce_options(options)?,
                    )]
                }
                crate::Operation::ReduceLogSum {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reduce_log_sum_with_options(
                        get_operand(&operands, *input),
                        &to_reduce_options(options)?,
                    )]
                }
                crate::Operation::ReduceLogSumExp {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reduce_log_sum_exp_with_options(
                        get_operand(&operands, *input),
                        &to_reduce_options(options)?,
                    )]
                }
                crate::Operation::ReduceSumSquare {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reduce_sum_square_with_options(
                        get_operand(&operands, *input),
                        &to_reduce_options(options)?,
                    )]
                }
                crate::Operation::Reshape {
                    input,
                    new_shape,
                    options:_,
                    outputs: _,
                } => {
                    vec![builder.reshape_with_options(
                        get_operand(&operands, *input),
                        &shape_to_mlshape(new_shape),
                        &ml_opts,
                    )]
                }
                crate::Operation::Resample2d {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.resample2d_with_options(
                        get_operand(&operands, *input),
                        &to_resample2d_options(options)?,
                    )]
                }
                crate::Operation::Reverse {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.reverse_with_options(
                        get_operand(&operands, *input),
                        &to_reverse_options(options)?,
                    )]
                }
                crate::Operation::ScatterElements {
                    input,
                    indices,
                    updates,
                    options,
                    outputs: _,
                } => {
                    vec![builder.scatter_elements_with_options(
                        get_operand(&operands, *input),
                        get_operand(&operands, *indices),
                        get_operand(&operands, *updates),
                        &to_scatter_options(options)?,
                    )]
                }
                crate::Operation::Softmax {
                    input,
                    axis,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.softmax_with_options(
                        get_operand(&operands, *input),
                        *axis,
                        &ml_opts,
                    )]
                }
                crate::Operation::Slice {
                    input,
                    starts,
                    sizes,
                    options,
                    outputs: _,
                } => {
                    vec![builder.slice_with_options(
                        get_operand(&operands, *input),
                        &u32_slice_to_js(starts),
                        &shape_to_mlshape(sizes),
                        &to_slice_options(options)?,
                    )]
                }
                crate::Operation::Split {
                    input,
                    splits,
                    split_equal_parts,
                    options,
                    outputs: _,
                } => {
                    if let Some(split_equal_parts) = split_equal_parts {
                        builder.split_with_u32_and_options(
                            get_operand(&operands, *input),
                            *split_equal_parts,
                            &to_split_options(options)?
                        )
                    } else {
                        builder.split_with_u32_sequence_and_options(
                            get_operand(&operands, *input),
                            &u32_slice_to_js(splits),
                            &to_split_options(options)?
                        )
                    }.to_vec()

                }
                crate::Operation::Transpose {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.transpose_with_options(
                        get_operand(&operands, *input),
                        &to_transpose_options(options)?
                    )]
                }
                crate::Operation::Squeeze {
                    input:_,
                    options: _,
                    outputs: _,
                } => {
                    todo!("not a webnn op")
                }
                crate::Operation::Unsqueeze {
                    input:_,
                    options: _,
                    outputs: _,
                } => todo!("not a webnn op"),
                crate::Operation::Tile {
                    input,
                    repetitions,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.tile_with_options(
                        get_operand(&operands, *input),
                        &u32_slice_to_js(repetitions),
                        &ml_opts
                    )]
                }
                crate::Operation::Triangular {
                    input,
                    options,
                    outputs: _,
                } => {
                    vec![builder.triangular_with_options(
                        get_operand(&operands, *input),
                        &to_triangular_options(options)?,
                    )]
                }
                crate::Operation::Prelu {
                    input,
                    slope,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.prelu_with_options(
                        get_operand(&operands, *input),
                        get_operand(&operands, *slope),
                        &ml_opts,
                    )]
                }
                crate::Operation::QuantizeLinear {
                    input,
                    scale,
                    zero_point,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.quantize_linear_with_options(
                        get_operand(&operands, *input),
                        get_operand(&operands, *scale),
                        get_operand(&operands, zero_point.expect("WebNN always expects a zero point!")),
                        &ml_opts,
                    )]
                }
                crate::Operation::DequantizeLinear {
                    input,
                    scale,
                    zero_point,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.dequantize_linear_with_options(
                        get_operand(&operands, *input),
                        get_operand(&operands, *scale),
                        get_operand(&operands, zero_point.expect("WebNN always expects a zero point!")),
                        &ml_opts,
                    )]
                }
                crate::Operation::Softplus {
                    input,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.softplus_with_options(
                        get_operand(&operands, *input),
                        &ml_opts,
                    )]
                }
                crate::Operation::Softsign {
                    input,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.softsign_with_options(
                        get_operand(&operands, *input),
                        &ml_opts,
                    )]
                }
                crate::Operation::Gelu {
                    input,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.gelu_with_options(
                        get_operand(&operands, *input),
                        &ml_opts,
                    )]
                }
                crate::Operation::Shape {
                    input: _,
                    options: _,
                    outputs: _,
                } => {
                    Err(WebNNConverterError::DynamicShapeNotSupported)?
                }
                crate::Operation::ScatterND {
                    input,
                    indices,
                    updates,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.scatter_nd_with_options(
                        get_operand(&operands, *input),
                        get_operand(&operands, *indices),
                        get_operand(&operands, *updates),
                        &ml_opts,
                    )]
                }
                crate::Operation::GatherND {
                    input,
                    indices,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.gather_nd_with_options(
                        get_operand(&operands, *input),
                        get_operand(&operands, *indices),
                        &ml_opts,
                    )]
                }
                crate::Operation::IsNaN {
                    input,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.is_na_n_with_options(
                        get_operand(&operands, *input),
                        &ml_opts,
                    )]
                }
                crate::Operation::IsInfinite {
                    input,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.is_infinite_with_options(
                        get_operand(&operands, *input),
                        &ml_opts,
                    )]
                }
                crate::Operation::RoundEven {
                    input,
                    options: _,
                    outputs: _,
                } => {
                    vec![builder.round_even_with_options(
                        get_operand(&operands, *input),
                        &ml_opts,
                    )]
                }
            };
            for (&k, v) in op.output_operands_slice().iter().zip(out.iter()) {
                let k = k as usize;
                if operands.len() >= k {
                    operands.resize(k + 1, None);
                }
                operands[k] = Some(v.clone());
            }
        }

        let output = js_sys::Object::new_typed();

        for (id, op) in graph_info.operands.iter().enumerate() {
            if op.kind == crate::OperandKind::Output {
                Reflect::set(
                    &output,
                    &operand_name(graph_info, id as u32).into(),
                    &get_operand(&operands, id as u32).clone(),
                )
                .unwrap();
            }
        }
        Ok(ConvertedGraph {
            format: "webnn",
            content_type: "application/octet-stream",
            data: vec![],
            weights_data: None,
            graph: Some(
                JsFuture::from(builder.build(&output))
                    .await
                    .map_err(WebNNConverterError::FailedToBuildGraph)?,
            ),
        })
    }
}
fn to_min_max_options(options: &Option<MLArgMinMaxOptions>) -> Result<MlArgMinMaxOptions> {
    let opts = MlArgMinMaxOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_keep_dimensions(options.keep_dimensions);
        opts.set_output_data_type(operand_dtype_to_dtype(options.output_data_type)?);
    }

    Ok(opts)
}
fn to_batch_norm_options(
    options: &Option<MLBatchNormalizationOptions>,
    operands: &[Option<MlOperand>],
) -> Result<MlBatchNormalizationOptions> {
    let opts = MlBatchNormalizationOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if let Some(scale) = options.scale {
            opts.set_scale(get_operand(operands, scale));
        }
        if let Some(bias) = options.bias {
            opts.set_bias(get_operand(operands, bias));
        }
        opts.set_axis(options.axis);
        opts.set_epsilon(options.epsilon);
    }

    Ok(opts)
}
fn to_clamp_options(options: &Option<MLClampOptions>) -> Result<MlClampOptions> {
    let opts = MlClampOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if let Some(serde_json::Value::Number(number)) = &options.min_value {
            opts.set_min_value_f64(
                number
                    .as_f64()
                    .ok_or_else(|| WebNNConverterError::FailedRepresentAsF64(number.clone()))?,
            )
        }
        if let Some(serde_json::Value::Number(number)) = &options.max_value {
            opts.set_max_value_f64(
                number
                    .as_f64()
                    .ok_or_else(|| WebNNConverterError::FailedRepresentAsF64(number.clone()))?,
            )
        }
    }

    Ok(opts)
}
//fn to_constant_options(options: &Option<MLConstantOptions>) -> Result<MLConstantOptions> {
//let opts = MLConstantOptions::new();
//if let Some(options) = options {
//opts.set_label(&options.label);
//opts.set_keep_dimensions(options.keep_dimensions);
//opts.set_output_data_type(str_dtype_to_dtype(&options.output_data_type)?);
//}

//Ok(opts)

//}
fn to_conv2d_options(
    options: &Option<MLConv2dOptions>,
    operands: &[Option<MlOperand>],
) -> Result<MlConv2dOptions> {
    let opts = MlConv2dOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if !options.padding.is_empty() {
            opts.set_padding(&u32_slice_to_js(&options.padding));
        }
        if !options.strides.is_empty() {
            opts.set_strides(&u32_slice_to_js(&options.strides));
        }
        if !options.dilations.is_empty() {
            opts.set_dilations(&u32_slice_to_js(&options.dilations));
        }
        opts.set_groups(options.groups);
        if let Some(bias) = options.bias {
            opts.set_bias(get_operand(operands, bias))
        }
        if !options.input_layout.is_empty() {
            opts.set_input_layout(to_input_layout(&options.input_layout)?);
        }
        if !options.filter_layout.is_empty() {
            opts.set_filter_layout(to_filter_operand_layout(&options.filter_layout)?);
        }
    }

    Ok(opts)
}
fn to_conv_transpose2d_options(
    options: &Option<MLConvTranspose2dOptions>,
    operands: &[Option<MlOperand>],
) -> Result<MlConvTranspose2dOptions> {
    let opts = MlConvTranspose2dOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if !options.padding.is_empty() {
            opts.set_padding(&u32_slice_to_js(&options.padding));
        }
        if !options.strides.is_empty() {
            opts.set_strides(&u32_slice_to_js(&options.strides));
        }
        if !options.dilations.is_empty() {
            opts.set_dilations(&u32_slice_to_js(&options.dilations));
        }
        opts.set_groups(options.groups);
        if let Some(bias) = options.bias {
            opts.set_bias(get_operand(operands, bias))
        }
        if !options.input_layout.is_empty() {
            opts.set_input_layout(to_input_layout(&options.input_layout)?);
        }
        if !options.output_padding.is_empty() {
            opts.set_output_padding(&u32_slice_to_js(&options.output_padding));
        }
        if let Some(output_sizes) = &options.output_sizes {
            opts.set_output_sizes(&u32_slice_to_js(output_sizes));
        }
        if !options.filter_layout.is_empty() {
            opts.set_filter_layout(to_filter_transpose_operand_layout(&options.filter_layout)?);
        }
    }

    Ok(opts)
}

fn to_cumulative_sum_options(
    options: &Option<MLCumulativeSumOptions>,
) -> Result<MlCumulativeSumOptions> {
    let opts = MlCumulativeSumOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_exclusive(options.exclusive);
        opts.set_reversed(options.reversed);
    }

    Ok(opts)
}

fn to_elu_options(options: &Option<MLEluOptions>) -> Result<MlEluOptions> {
    let opts = MlEluOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_alpha(options.alpha);
    }

    Ok(opts)
}

fn to_gather_options(options: &Option<MLGatherOptions>) -> Result<MlGatherOptions> {
    let opts = MlGatherOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_axis(options.axis);
    }

    Ok(opts)
}

fn to_gemm_options(options: &Option<MLGemmOptions>) -> Result<MlGemmOptions> {
    let opts = MlGemmOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_alpha(options.alpha);
        opts.set_a_transpose(options.a_transpose);
        opts.set_beta(options.beta);
        opts.set_b_transpose(options.b_transpose);
    }

    Ok(opts)
}

fn to_gru_options(
    options: &Option<MLGruOptions>,
    operands: &[Option<MlOperand>],
) -> Result<MlGruOptions> {
    let opts = MlGruOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if let Some(bias) = options.bias {
            opts.set_bias(get_operand(operands, bias));
        }
        if let Some(recurrent_bias) = options.recurrent_bias {
            opts.set_recurrent_bias(get_operand(operands, recurrent_bias));
        }
        if let Some(initial_hidden_state) = options.initial_hidden_state {
            opts.set_initial_hidden_state(get_operand(operands, initial_hidden_state));
        }
        opts.set_reset_after(options.reset_after);
        opts.set_return_sequence(options.return_sequence);
        opts.set_direction(to_recurrent_network_direction(&options.direction)?);
        if let Some(activations) = &options.activations {
            opts.set_activations(
                &activations
                    .iter()
                    .map(|a| a.as_str().into())
                    .collect::<Vec<_>>(),
            )
        }
        if !options.layout.is_empty() {
            opts.set_layout(to_gru_weight_layout(&options.layout)?);
        }
    }

    Ok(opts)
}

fn to_gru_cell_options(
    options: &Option<MLGruCellOptions>,
    operands: &[Option<MlOperand>],
) -> Result<MlGruCellOptions> {
    let opts = MlGruCellOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if let Some(bias) = options.bias {
            opts.set_bias(get_operand(operands, bias));
        }
        if let Some(recurrent_bias) = options.recurrent_bias {
            opts.set_recurrent_bias(get_operand(operands, recurrent_bias));
        }
        opts.set_reset_after(options.reset_after);
        if let Some(activations) = &options.activations {
            opts.set_activations(
                &activations
                    .iter()
                    .map(|a| a.as_str().into())
                    .collect::<Vec<_>>(),
            )
        }
        if !options.layout.is_empty() {
            opts.set_layout(to_gru_weight_layout(&options.layout)?);
        }
    }

    Ok(opts)
}

fn to_hard_sigmoid_options(options: &Option<MLHardSigmoidOptions>) -> Result<MlHardSigmoidOptions> {
    let opts = MlHardSigmoidOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_alpha(options.alpha);
        opts.set_beta(options.beta);
    }

    Ok(opts)
}

fn to_instance_normalization_options(
    options: &Option<MLInstanceNormalizationOptions>,
    operands: &[Option<MlOperand>],
) -> Result<MlInstanceNormalizationOptions> {
    let opts = MlInstanceNormalizationOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if let Some(scale) = options.scale {
            opts.set_scale(get_operand(operands, scale));
        }
        if let Some(bias) = options.bias {
            opts.set_bias(get_operand(operands, bias));
        }
        opts.set_epsilon(options.epsilon);
        if !options.layout.is_empty() {
            opts.set_layout(to_input_layout(&options.layout)?)
        }
    }

    Ok(opts)
}

fn to_layer_normalization_options(
    options: &Option<MLLayerNormalizationOptions>,
    operands: &[Option<MlOperand>],
) -> Result<MlLayerNormalizationOptions> {
    let opts = MlLayerNormalizationOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if let Some(scale) = options.scale {
            opts.set_scale(get_operand(operands, scale));
        }
        if let Some(bias) = options.bias {
            opts.set_bias(get_operand(operands, bias));
        }
        opts.set_epsilon(options.epsilon);
        if let Some(axes) = &options.axes {
            opts.set_axes(&u32_slice_to_js(axes));
        }
    }

    Ok(opts)
}

fn to_leaky_relu_options(options: &Option<MLLeakyReluOptions>) -> Result<MlLeakyReluOptions> {
    let opts = MlLeakyReluOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_alpha(options.alpha);
    }

    Ok(opts)
}

fn to_linear_options(options: &Option<MLLinearOptions>) -> Result<MlLinearOptions> {
    let opts = MlLinearOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_alpha(options.alpha);
        opts.set_beta(options.beta);
    }

    Ok(opts)
}

fn to_pad_options(options: &Option<MLPadOptions>) -> Result<MlPadOptions> {
    let opts = MlPadOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_mode(to_padding_mode(&options.mode)?);
    }

    Ok(opts)
}

fn to_pool2d_options(options: &Option<MLPool2dOptions>) -> Result<MlPool2dOptions> {
    let opts = MlPool2dOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if let Some(window_dimensions) = &options.window_dimensions {
            opts.set_window_dimensions(&u32_slice_to_js(window_dimensions));
        }
        opts.set_padding(&u32_slice_to_js(&options.padding));
        opts.set_strides(&u32_slice_to_js(&options.strides));
        opts.set_dilations(&u32_slice_to_js(&options.dilations));
        if !options.layout.is_empty() {
            opts.set_layout(to_input_layout(&options.layout)?);
        }
        opts.set_output_shape_rounding(to_rounding_type(&options.output_shape_rounding)?);
        if let Some(output_sizes) = &options.output_sizes {
            opts.set_output_sizes(&u32_slice_to_js(output_sizes));
        }
    }

    Ok(opts)
}

fn to_reduce_options(options: &Option<MLReduceOptions>) -> Result<MlReduceOptions> {
    let opts = MlReduceOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if let Some(axes) = &options.axes {
            opts.set_axes(&u32_slice_to_js(axes));
        }
        opts.set_keep_dimensions(options.keep_dimensions);
    }

    Ok(opts)
}

fn to_resample2d_options(options: &Option<MLResample2dOptions>) -> Result<MlResample2dOptions> {
    let opts = MlResample2dOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_mode(to_resample2d_mode(&options.mode)?);
        opts.set_scales(&f32_slice_to_js(&options.scales));
        if let Some(sizes) = options.sizes.as_ref() {
            opts.set_sizes(&u32_slice_to_js(sizes));
        }
        opts.set_axes(&u32_slice_to_js(&options.axes));
    }

    Ok(opts)
}

fn to_reverse_options(options: &Option<MLReverseOptions>) -> Result<MlReverseOptions> {
    let opts = MlReverseOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        if let Some(axes) = options.axes.as_ref() {
            opts.set_axes(&u32_slice_to_js(axes));
        }
    }

    Ok(opts)
}

fn to_scatter_options(options: &Option<MLScatterOptions>) -> Result<MlScatterOptions> {
    let opts = MlScatterOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_axis(options.axis);
    }

    Ok(opts)
}

fn to_slice_options(options: &Option<MLSliceOptions>) -> Result<MlSliceOptions> {
    let opts = MlSliceOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        // TODO: hack: to work around wrong defaults of RustNN
        // must be same rank as operand
        if !options.strides.is_empty() {
            opts.set_strides(&u32_slice_to_js(&options.strides));
        }
    }

    Ok(opts)
}

fn to_split_options(options: &Option<MLSplitOptions>) -> Result<MlSplitOptions> {
    let opts = MlSplitOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_axis(options.axis);
    }

    Ok(opts)
}

fn to_transpose_options(options: &Option<MLTransposeOptions>) -> Result<MlTransposeOptions> {
    let opts = MlTransposeOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);

        // TODO: hack: to work around wrong defaults of RustNN
        // default for permutation is identity permutation of same rank as input operand
        if !options.permutation.is_empty() {
            opts.set_permutation(&u32_slice_to_js(&options.permutation));
        }
    }

    Ok(opts)
}

fn to_triangular_options(options: &Option<MLTriangularOptions>) -> Result<MlTriangularOptions> {
    let opts = MlTriangularOptions::new();
    if let Some(options) = options {
        opts.set_label(&options.label);
        opts.set_upper(options.upper.is_some_and(|u| u));
        opts.set_diagonal(options.diagonal);
    }

    Ok(opts)
}

fn dtype_to_dtype(dtype: DataType) -> Result<MlOperandDataType> {
    Ok(match dtype {
        DataType::Float16 => MlOperandDataType::Float16,
        DataType::Float32 => MlOperandDataType::Float32,
        DataType::Int32 => MlOperandDataType::Int32,
        DataType::Uint32 => MlOperandDataType::Uint32,
        DataType::Int8 => MlOperandDataType::Int8,
        DataType::Uint8 => MlOperandDataType::Uint8,
        DataType::Int64 => MlOperandDataType::Int64,
        DataType::Uint64 => MlOperandDataType::Uint64,
        DataType::Int4 | DataType::Uint4 => {
            return Err(WebNNConverterError::UnsupportedDataType(dtype))?;
        }
    })
}

fn operand_dtype_to_dtype(dtype: MLOperandDataType) -> Result<MlOperandDataType> {
    Ok(match dtype {
        MLOperandDataType::Float32 => MlOperandDataType::Float32,
        MLOperandDataType::Float16 => MlOperandDataType::Float16,
        MLOperandDataType::Int32 => MlOperandDataType::Int32,
        MLOperandDataType::Uint32 => MlOperandDataType::Uint32,
        MLOperandDataType::Int8 => MlOperandDataType::Int8,
        MLOperandDataType::Uint8 => MlOperandDataType::Uint8,
        MLOperandDataType::Int64 => MlOperandDataType::Int64,
        MLOperandDataType::Uint64 => MlOperandDataType::Uint64,
    })
}

fn to_input_layout(input_layout: &str) -> Result<MlInputOperandLayout> {
    Ok(match input_layout {
        "nchw" => MlInputOperandLayout::Nchw,
        "nhwc" => MlInputOperandLayout::Nhwc,
        _ => {
            return Err(GraphError::ConversionFailed {
                format: "webnn".to_string(),
                reason: format!("Unsupported input layout: {:?}", input_layout),
            });
        }
    })
}

fn to_rounding_type(rounding_type: &str) -> Result<MlRoundingType> {
    Ok(match rounding_type {
        "floor" => MlRoundingType::Floor,
        "ceil" => MlRoundingType::Ceil,
        _ => {
            return Err(GraphError::ConversionFailed {
                format: "webnn".to_string(),
                reason: format!("Unsupported rounding type: {rounding_type:?}"),
            });
        }
    })
}

fn to_recurrent_network_direction(input_layout: &str) -> Result<MlRecurrentNetworkDirection> {
    Ok(match input_layout {
        "forward" => MlRecurrentNetworkDirection::Forward,
        "backward" => MlRecurrentNetworkDirection::Backward,
        "both" => MlRecurrentNetworkDirection::Both,
        _ => {
            return Err(GraphError::ConversionFailed {
                format: "webnn".to_string(),
                reason: format!(
                    "Unsupported recurrent network direction: {:?}",
                    input_layout
                ),
            });
        }
    })
}

#[allow(dead_code)]
fn to_recurrent_network_activation(activation: &str) -> Result<MlRecurrentNetworkActivation> {
    Ok(match activation {
        "relu" => MlRecurrentNetworkActivation::Relu,
        "sigmoid" => MlRecurrentNetworkActivation::Sigmoid,
        "tanh" => MlRecurrentNetworkActivation::Tanh,
        _ => {
            return Err(GraphError::ConversionFailed {
                format: "webnn".to_string(),
                reason: format!("Unsupported recurrent network activation: {:?}", activation),
            });
        }
    })
}

fn to_gru_weight_layout(layout: &str) -> Result<MlGruWeightLayout> {
    Ok(match layout {
        "zrn" => MlGruWeightLayout::Zrn,
        "rzn" => MlGruWeightLayout::Rzn,
        _ => {
            return Err(GraphError::ConversionFailed {
                format: "webnn".to_string(),
                reason: format!("Unsupported gru weight layout: {:?}", layout),
            });
        }
    })
}

fn to_filter_operand_layout(filter_operand_layout: &str) -> Result<MlConv2dFilterOperandLayout> {
    Ok(match filter_operand_layout {
        "oihw" => MlConv2dFilterOperandLayout::Oihw,
        "hwio" => MlConv2dFilterOperandLayout::Hwio,
        "ohwi" => MlConv2dFilterOperandLayout::Ohwi,
        "ihwo" => MlConv2dFilterOperandLayout::Ihwo,
        _ => {
            return Err(GraphError::ConversionFailed {
                format: "webnn".to_string(),
                reason: format!(
                    "Unsupported filter operand layout: {:?}",
                    filter_operand_layout
                ),
            });
        }
    })
}

fn to_filter_transpose_operand_layout(
    filter_operand_layout: &str,
) -> Result<MlConvTranspose2dFilterOperandLayout> {
    Ok(match filter_operand_layout {
        "iohw" => MlConvTranspose2dFilterOperandLayout::Iohw,
        "hwoi" => MlConvTranspose2dFilterOperandLayout::Hwoi,
        "ohwi" => MlConvTranspose2dFilterOperandLayout::Ohwi,
        _ => {
            return Err(GraphError::ConversionFailed {
                format: "webnn".to_string(),
                reason: format!(
                    "Unsupported filter operand layout: {:?}",
                    filter_operand_layout
                ),
            });
        }
    })
}

fn to_padding_mode(mode: &str) -> Result<MlPaddingMode> {
    Ok(match mode {
        "constant" => MlPaddingMode::Constant,
        "edge" => MlPaddingMode::Edge,
        "reflection" => MlPaddingMode::Reflection,
        _ => {
            return Err(GraphError::ConversionFailed {
                format: "webnn".to_string(),
                reason: format!("Unsupported padding mode: {mode:?}"),
            });
        }
    })
}

fn to_resample2d_mode(mode: &str) -> Result<MlInterpolationMode> {
    Ok(match mode {
        "nearest-neighbor" => MlInterpolationMode::NearestNeighbor,
        "linear" => MlInterpolationMode::Linear,
        _ => {
            return Err(GraphError::ConversionFailed {
                format: "webnn".to_string(),
                reason: format!("Unsupported interpolation mode: {mode:?}"),
            });
        }
    })
}

fn shape_to_mlshape(shape: &[MLDimension]) -> Vec<js_sys::Number> {
    shape.iter().map(|s| s.static_or_max().into()).collect()
}

fn shape_to_shape(shape: &[Dimension]) -> Vec<js_sys::Number> {
    shape
        .iter()
        .map(|s| s.get_static_or_max_size().into())
        .collect()
}

impl crate::converters::GraphConverter for WebNNConverter {
    fn format(&self) -> &'static str {
        "webnn"
    }

    fn convert(&self, graph_info: &GraphInfo) -> Result<ConvertedGraph> {
        let context: MlContext = JsFuture::from(self.context.clone())
            .block_on()
            .map_err(|_| WebNNConverterError::FailedToCreateMlContext)?;
        self.convert_async(&context, graph_info).block_on()
    }
}

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2023. All rights reserved.
 * Description: nn_defs
 */
#ifndef INC_API_GRAPH_OP_NN_DEFS_H
#define INC_API_GRAPH_OP_NN_DEFS_H
#include "graph/operator_hiai_reg.h"

// clang-format off
namespace hiai {
/*
 * The Activation op provides different types of nonlinearities for use in neural networks.
 * These include smooth nonlinearities (sigmoid, tanh, and softplus), continuous but not everywhere
 * differentiable functions (ReLU), and random regularization.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor with the same type as the input tensor
 * <Attr>
 *    mode : Activation mode, with options as follows:
 *           0 : Sigmoid
 *           1 : ReLU
 *           2 : Tanh
 *           3 : Clipped ReLU
 *           4 : ELU
 *           5 : LeakyRelu
 *           6 : Abs
 *           7 : Relu1
 *           8 : Softsign
 *           9 : Softplus
 *           10 : Hardsigmoid
 *           11 : Threshold ReLU
 *           12 : Selu
 *           14 : Relu6
 *           15 : GeLU.
 *           Defaults to 1 (ReLU). 1.
 *    coef : If mode is set to Clipped ReLU(3), ELU(4), Hardsigmoid(10), Threshold ReLU(11), coef can be specified.
 *           otherwise, the value of coef must use default value.
 *    negative_slope : If mode is set to Leaky_Relu(5), negative_slope can be specified.
 *                     otherwise, the value of negative_slope must use default value.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    Clipped ReLU:
 *    auto activation = hiai::op::Activation("activation")
 *                      .set_input_x(x)
 *                      .set_attr_mode(3)
 *                      .set_attr_coef(6.0);
 *
 *    ELU:
 *    auto activation = hiai::op::Activation("activation")
 *                      .set_input_x(x)
 *                      .set_attr_mode(4)
 *                      .set_attr_coef(1.0);
 *
 *    PReLU:
 *    auto activation = hiai::op::Activation("activation")
 *                      .set_input_x(x)
 *                      .set_attr_mode(5)
 *                      .set_attr_negative_slope(0.5);
 *
 *    Threshold ReLU:
 *    auto activation = hiai::op::Activation("activation")
 *                      .set_input_x(x)
 *                      .set_attr_mode(11)
 *                      .set_attr_coef(0.1);
 *
 *    Other mode:
 *    auto activation = hiai::op::Activation("activation")
 *                      .set_input_x(x)
 *                      .set_attr_mode(0);
 */
HIAI_REG_OP(Activation)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(mode, AttrValue::INT { 1 })
.HIAI_ATTR(coef, AttrValue::FLOAT { 0.0 })
.HIAI_ATTR(negative_slope, AttrValue::FLOAT { 0.0 })
.HIAI_OP_END()

/*
 * Computes hard-swish activation function.
 * <Input>
 *    x : The input tensor.
 * <Output>
 *    y : The output tensor.
 * <Added in HiAI version>
 *    100.320.010.022
 */
HIAI_REG_OP(HardSwish)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Normalizes the input to have 0-mean and/or unit (1) variance across the batch (batch normalization).
 * <Input>
 *    x : Input tensor
 *    mean : Tensor for population mean. Used for inference only.
 *    variance : Tensor for population variance. Used for inference only.
 *    scale : Tensor for scaling factor, to scale the normalized 'x'
 *    offset : Tensor for bias, to shift to the normalized 'x'
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    momentum : Momentum for the running mean and running variance
 *               running_mean is equal to  [ running_mean * momentum + mean * (1 - momentum)].
 *    epsilon : Small float number added to the variance of 'x'
 *    mode : BatchNorm mode.
 *           0: The bnScale and bnBias tensors are of size 1xCxHxW.
 *           1: The bnScale and bnBias tensors are of size 1xCx1x1.
 *    use_global_stats : Must be true.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc meanTensorDesc(Shape({1, 5, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr meanTensor = std::make_shared<hiai::Tensor>(meanTensorDesc);
 *    vector<float> meanDataValue(5, 0.0);
 *    meanTensor->SetData((uint8_t*)meanDataValue.data(), 5 * sizeof(float));
 *    auto mean = hiai::op::Const("mean").set_attr_value(meanTensor);
 *
 *    TensorDesc varianceTensorDesc(Shape({1, 5, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr varianceTensor = std::make_shared<hiai::Tensor>(varianceTensorDesc);
 *    vector<float> varianceDataValue(5, 0.0);
 *    varianceTensor->SetData((uint8_t*)varianceDataValue.data(), 5 * sizeof(float));
 *    auto variance = hiai::op::Const("variance").set_attr_value(varianceTensor);
 *
 *    TensorDesc scaleTensorDesc(Shape({1, 5, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr scaleTensor = std::make_shared<hiai::Tensor>(scaleTensorDesc);
 *    vector<float> scaleDataValue(5, 0.0);
 *    scaleTensor->SetData((uint8_t*)scaleDataValue.data(), 5 * sizeof(float));
 *    auto scale = hiai::op::Const("scale").set_attr_value(scaleTensor);
 *
 *    TensorDesc offsetTensorDesc(Shape({1, 5, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr offsetTensor = std::make_shared<hiai::Tensor>(offsetTensorDesc);
 *    vector<float> offsetDataValue(5, 0.0);
 *    offsetTensor->SetData((uint8_t*)offsetDataValue.data(), 5 * sizeof(float));
 *    auto offset = hiai::op::Const("offset").set_attr_value(offsetTensor);
 *
 *    auto bnInference = hiai::op::BNInference("bnInference")
 *                       .set_input_x(x)
 *                       .set_input_mean(mean)
 *                       .set_input_variance(variance)
 *                       .set_input_scale(scale)
 *                       .set_input_offset(offset)
 *                       .set_attr_mode(1);
 */
HIAI_REG_OP(BNInference)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(mean, TensorType({ DT_FLOAT }))
.HIAI_INPUT(variance, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_INPUT(scale, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_INPUT(offset, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(momentum, AttrValue::FLOAT { 0.9f })
.HIAI_ATTR(epsilon, AttrValue::FLOAT { 1e-5f })
.HIAI_ATTR(mode, AttrValue::INT { 1 })
.HIAI_ATTR(use_global_stats, AttrValue::BOOL { true })
.HIAI_OP_END()

/*
 * Consumes an input tensor and a filter, and computes the output.
 * <Input>
 *    x : Input tensor with size [N, Ci, Hi, Wi] or [N, Ci, Wi] or [N, Hi, Wi, Ci] or [N, Wi, Ci].
 *    filter : With shape [Co, Ci/group, Hk, Wk] or [Co, Ci/group, Wk], must be a const op when data_format is 'NHWC'.
 *    bias : With shape [Co].
 *    offset_w : Reserved. For quantized.
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    strides : Stride along each axis.
 *    dilations : Dilation value along each axis of the filter.
 *    pads : Padding for the beginning and ending along each axis [hh, ht, wh, wt] or [wh, wt].
 *    pad_mode : Pad mode, SPECIFIC(Default): not set, using pads; SAME, SAME_UPPER, SAME_LOWER or VALID.
 *               The padding is split between the two sides equally or
 *               almost equally (depending on whether it is even or odd).
 *               In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER or SAME and
 *               at the beginning for SAME_LOWER
 *    groups : Number of groups input channels and output channels are divided into.
 *             When groups = 1, traditional convolution will be performed;
 *             When groups > 1, feature maps are grouped by group_count, and then each groups
 *             is convoluted separately. Specially, 'groups' equal to the number of input feature
 *             maps indicates Depthwise Convolution.
 *    data_format : Format of operator, 'NCHW' or 'NHWC'. Default is 'NCHW'
 *    offset_x : Reserved. For quantized.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc filterTensorDesc(Shape({7, 5, 3, 4}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr filterTensor = std::make_shared<hiai::Tensor>(filterTensorDesc);
 *    vector<float> filterDataValue(7 * 5 * 3 * 4, 0);
 *    filterTensor->SetData((uint8_t*)filterDataValue.data(), 7 * 5 * 3 * 4 * sizeof(float));
 *    auto filter = hiai::op::Const("filter").set_attr_value(filterTensor);
 *
 *    TensorDesc biasTensorDesc(Shape({7}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr biasTensor = std::make_shared<hiai::Tensor>(biasTensorDesc);
 *    vector<float> biasDataValue(7, 0);
 *    biasTensor->SetData((uint8_t*)biasDataValue.data(), 7 * sizeof(float));
 *    auto bias = hiai::op::Const("bias").set_attr_value(biasTensor);
 *
 *    auto convolution = hiai::op::Convolution("convolution")
 *                       .set_input_x(x)
 *                       .set_input_filter(filter)
 *                       .set_input_bias(bias)
 *                       .set_attr_strides({1, 1})
 *                       .set_attr_dilations({1, 1})
 *                       .set_attr_pads({0, 0, 0, 1})
 *                       .set_attr_pad_mode("SAME")
 *                       .set_attr_groups(1)
 *                       .set_attr_data_format("NCHW");
 */
HIAI_REG_OP(Convolution)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_UINT8 }))
.HIAI_INPUT(filter, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_OPTIONAL_INPUT(bias, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OPTIONAL_INPUT(offset_w, TensorType({ DT_INT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_REQUIRED_ATTR(strides, AttrValue::LIST_INT)
.HIAI_ATTR(dilations, AttrValue::LIST_INT ({ 1, 1 }))
.HIAI_ATTR(pads, AttrValue::LIST_INT ({ 0, 0, 0, 0 }))
.HIAI_ATTR(pad_mode, AttrValue::STR { "SPECIFIC" })
.HIAI_ATTR(groups, AttrValue::INT { 1 })
.HIAI_ATTR(data_format, AttrValue::STR { "NCHW" })
.HIAI_ATTR(offset_x, AttrValue::INT { 0 })
.HIAI_OP_END()

/*
 * Quantized for Convolution, consumes an input tensor and a filter, and computes the output.
 * <Input>
 *    x : Input tensor with size [N, Ci, Hi, Wi].
 *    filter : With shape [Co, Ci/group, Hk, Wk], must be a Const-OP.
 *             If filter_quant_type = 1 or 2, shall use type 'tensor(int8)'.
 *             for(i in C) { quant_filter[i] = origin_filter[i] / filter_quant_scales[i];}
 *    bias : With shape [Co], must be a const OP.
 *           If filter_quant_type = 1 or 2, shall use type 'tensor(int32)'.
 *           for(i in C) { quant_bias[i] = origin_bias[i] / (x_quant_scale * filter_quant_scales[i]);}
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    strides : Stride along each axis.
 *    dilations : Dilation value along each axis of the filter.
 *    pads : Padding for the beginning and ending along each axis [hh, ht, wh, wt].
 *    pad_mode : Pad mode, SPECIFIC(Default): not set, using pads;
 *               SAME, or VALID: indicate the type of padding algorithm instead of using parameter "pads".
 *    groups : Number of groups input channels and output channels are divided into.
 *             When groups = 1, traditional convolution will be performed;
 *             When groups > 1, feature maps are grouped by group_count, and then each groups
 *             is convoluted separately. Specially, 'groups' equal to the number of input feature
 *             maps indicates Depthwise Convolution.
 *    data_format : Format of operator, 'NCHW' or 'NHWC'. Default is 'NCHW'
 *    x_quant_type : Either 0-No quant or 1-Quant8 (linear) or 2-Quant4 (linear); Reserved: 3-Quant2 (linear)
 *    filter_quant_type : Either 0-No quant or 1-Quant8 (linear) or 2-Quant4 (linear); Reserved: 3-Quant2 (linear)
 *    x_quant_scale : If x_quant_type > 0, this Attr is required.
 *    x_quant_offset : Data offset, if x_quant_type = 2, this Attr value should be zero.
 *    filter_quant_scales : List of scale factors of the filter. If filter_quant_type > 0, this Attr is required.
 * <Added in HiAI version>
 *    100.310.010.015
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc filterTensorDesc(Shape({7, 5, 3, 4}), FORMAT_NCHW, DT_INT8);
 *    TensorPtr filterTensor = std::make_shared<hiai::Tensor>(filterTensorDesc);
 *    vector<int8_t> filterDataValue(7 * 5 * 3 * 4, 0);
 *    filterTensor->SetData((uint8_t*)filterDataValue.data(), 7 * 5 * 3 * 4 * sizeof(int8_t));
 *    auto filter = hiai::op::Const("filter").set_attr_value(filterTensor);
 *
 *    TensorDesc biasTensorDesc(Shape({7}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr biasTensor = std::make_shared<hiai::Tensor>(biasTensorDesc);
 *    vector<int32_t> biasDataValue(7, 0);
 *    biasTensor->SetData((uint8_t*)biasDataValue.data(), 7 * sizeof(int32_t));
 *    auto bias = hiai::op::Const("bias").set_attr_value(biasTensor);
 *
 *    std::vector<ge::AttrValue::FLOAT> quantScales(7, 1.0);
 *    auto quantizedConvolution = hiai::op::QuantizedConvolution("quantizedConvolution")
 *                                .set_input_x(x)
 *                                .set_input_filter(filter)
 *                                .set_input_bias(bias)
 *                                .set_attr_strides({1, 1})
 *                                .set_attr_dilations({1, 1})
 *                                .set_attr_pads({0, 0, 0, 1})
 *                                .set_attr_pad_mode("SAME")
 *                                .set_attr_groups(1)
 *                                .set_attr_data_format("NCHW")
 *                                .set_attr_x_quant_type(1)
 *                                .set_attr_filter_quant_type(1)
 *                                .set_attr_x_quant_scale(1.0)
 *                                .set_attr_x_quant_offset(1.0)
 *                                .set_attr_filter_quant_scales(quantScales);
 */
HIAI_REG_OP(QuantizedConvolution)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(filter, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_OPTIONAL_INPUT(bias, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(strides, AttrValue::LIST_INT)
.HIAI_ATTR(dilations, AttrValue::LIST_INT ({ 1, 1 }))
.HIAI_ATTR(pads, AttrValue::LIST_INT ({ 0, 0, 0, 0 }))
.HIAI_ATTR(pad_mode, AttrValue::STR { "SPECIFIC" })
.HIAI_ATTR(groups, AttrValue::INT { 1 })
.HIAI_ATTR(data_format, AttrValue::STR { "NCHW" })
.HIAI_ATTR(x_quant_type, AttrValue::INT { 0 })
.HIAI_ATTR(filter_quant_type, AttrValue::INT { 0 })
.HIAI_ATTR(x_quant_scale, AttrValue::FLOAT { 1.0 })
.HIAI_ATTR(x_quant_offset, AttrValue::INT { 0 })
.HIAI_ATTR(filter_quant_scales, AttrValue::LIST_FLOAT ({}))
.HIAI_OP_END()

/*
 * Computes the gradients of convolution with respect to the output_shape.
 * If the HiaiVersion is earlier than 320, the input output_shape is required and the bias is not supported.
 * <Input>
 *    output_shape : The output shape, which is [batch, output_channels, height, width] or
 *                   [batch, output_channels, width] tensor
 *    filter : when data_format is 'NCHW', with shape [Ci, Co/group, Hk, Wk] or [Ci, Co/group, Wk]
 *             must be a Const-OP.
 *    x : Input tensor with size [N, Ci, Hi, Wi] or [N, Ci, Wi].
 *    bias : With shape [Co], must be a Const-OP.
 *    offset_w : Reserved. For quantized.
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    strides : Stride along each axis.
 *    pads : Padding for the beginning and ending along each axis [hh, ht, wh, wt] or [wh, wt].
 *    pad_mode : Pad mode, SPECIFIC(Default): not set, using pads; SAME, or VALID
 *    dilations : Dilation value along each axis of the filter.
 *    groups : Number of groups input channels and output channels are divided into.
 *             When groups = 1, traditional convolution will be performed;
 *             When groups > 1, feature maps are grouped by group_count, and then each groups
 *             is convoluted separately. Specially, 'groups' equal to the number of input feature
 *             maps indicates Depthwise Convolution.
 *    data_format : Format of operator, 'NCHW' or 'NHWC'. Default is 'NCHW'
 *    offset_x : Reserved. For quantized.
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc outputShapeTensorDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr outputShapeTensor = std::make_shared<hiai::Tensor>(outputShapeTensorDesc);
 *    vector<int32_t> outputShapeDataValue = {5, 6, 7, 8} ;
 *    outputShapeTensor->SetData((uint8_t*)outputShapeDataValue.data(), 4 * sizeof(int32_t));
 *    auto outputShape = hiai::op::Const("outputShape").set_attr_value(outputShapeTensor);
 *
 *    TensorDesc filterTensorDesc(Shape({5, 5, 3, 4}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr filterTensor = std::make_shared<hiai::Tensor>(filterTensorDesc);
 *    vector<float> filterDataValue(5 * 5 * 3 * 4, 0);
 *    filterTensor->SetData((uint8_t*)filterDataValue.data(), 5 * 5 * 3 * 4 * sizeof(float));
 *    auto filter = hiai::op::Const("filter").set_attr_value(filterTensor);
 *
 *    auto convTranspose = hiai::op::ConvTranspose("convTranspose")
 *                         .set_input_output_shape(outputShape)
 *                         .set_input_filter(filter)
 *                         .set_input_x(x)
 *                         .set_attr_strides({1, 1})
 *                         .set_attr_dilations({1, 1})
 *                         .set_attr_pads({0, 0, 0, 1})
 *                         .set_attr_pad_mode("SAME")
 *                         .set_attr_groups(1)
 *                         .set_attr_data_format("NCHW");
 */
HIAI_REG_OP(ConvTranspose)
.HIAI_OPTIONAL_INPUT(output_shape, TensorType({ DT_INT32 }))
.HIAI_INPUT(filter, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT16, DT_UINT8 }))
.HIAI_OPTIONAL_INPUT(bias, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OPTIONAL_INPUT(offset_w, TensorType({ DT_INT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_REQUIRED_ATTR(strides, AttrValue::LIST_INT)
.HIAI_ATTR(pads, AttrValue::LIST_INT ({ 0, 0, 0, 0 }))
.HIAI_ATTR(pad_mode, AttrValue::STR { "SPECIFIC" })
.HIAI_ATTR(dilations, AttrValue::LIST_INT ({ 1, 1 }))
.HIAI_ATTR(groups, AttrValue::INT { 1 })
.HIAI_ATTR(data_format, AttrValue::STR { "NCHW" })
.HIAI_ATTR(offset_x, AttrValue::INT { 0 })
.HIAI_OP_END()

/*
 * Adds bias to 'x'.
 * <Input>
 *    x : Input tensor of any number of dimensions
 *    bias : 1D tensor of size equal to the C dimension of 'x'
 * <Output>
 *    y : Broadcast sum of 'x' and 'bias'
 * <Attr>
 *    data_format : Format of operator, 'NCHW' or 'NHWC'. Default is 'NCHW'
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc biasTensorDesc(Shape({5}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr biasTensor = std::make_shared<hiai::Tensor>(biasTensorDesc);
 *    vector<float> dataValue(5, 0.0);
 *    biasTensor->SetData((uint8_t*)dataValue.data(), 5 * sizeof(float));
 *    auto bias = hiai::op::Const("bias").set_attr_value(biasTensor);
 *
 *    auto biasAdd = hiai::op::BiasAdd("biasAdd")
 *                   .set_input_x(x)
 *                   .set_input_bias(bias);
 */
HIAI_REG_OP(BiasAdd)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(bias, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(data_format, AttrValue::STR { "NCHW" })
.HIAI_OP_END()

/*
 * Compute all input tensors element-wise with specific mode.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    N : The count of the x.
 *    mode : Either 0 (product), 1 (sum), 2 (max). Defaults to 1 (sum).
 *    coeff : Blob-wise coefficient for sum operation. coeff can be set only when mode is set to 1.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc1(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x1 = hiai::op::Data("x1");
 *    x1.update_input_desc_x(xDesc1);
 *
 *    TensorDesc xDesc2(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x2 = hiai::op::Data("x2");
 *    x2.update_input_desc_x(xDesc2);
 *
 *    auto eltwise = hiai::op::Eltwise("eltwise")
 *                   .create_dynamic_input_x(2)
 *                   .set_dynamic_input_x(1, x1)
 *                   .set_dynamic_input_x(2, x2)
 *                   .set_attr_N(2)
 *                   .set_attr_mode(0);
 */
HIAI_REG_OP(Eltwise)
.HIAI_DYNAMIC_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(N, AttrValue::INT)
.HIAI_ATTR(mode, AttrValue::INT { 1 })
.HIAI_ATTR(coeff, AttrValue::LIST_FLOAT ({}))
.HIAI_OP_END()

/*
 * Local Response Normalization.
 * sqr_sum[a, b, c, d] = sum(input[a, b - ((depth_radius - 1 ) / 2) : b + ((depth_radius - 1 ) / 2) + 1, c, d ] ** 2).
 * output = input / (bias + (alpha / depth_radius) * sqr_sum) ** beta.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    depth_radius : Number of channels to sum over. is an odd number.
 *    bias : Offset (usually positive to avoid dividing by 0).
 *    alpha : Scale factor, usually positive.
 *    beta : Exponent value.
 *    norm_region : WITHIN_CHANNEL or ACROSS_CHANNELS. default:ACROSS_CHANNELS. Normalization between channels.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto lrn = hiai::op::LRN("lrn")
 *               .set_input_x(x)
 *               .set_attr_depth_radius(5)
 *               .set_attr_bias(1.0)
 *               .set_attr_alpha(1.0)
 *               .set_attr_beta(0.5);
 */
HIAI_REG_OP(LRN)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(depth_radius, AttrValue::INT { 5 })
.HIAI_ATTR(bias, AttrValue::FLOAT { 1.0 })
.HIAI_ATTR(alpha, AttrValue::FLOAT { 1.0 })
.HIAI_ATTR(beta, AttrValue::FLOAT { 0.5 })
.HIAI_ATTR(norm_region, AttrValue::STR { "ACROSS_CHANNELS" })
.HIAI_OP_END()

/*
 * Computes a depthwise convolution from given input and filter tensors.
 * <Input>
 *    x : Input tensor with size [N, Ci, Hi, Wi].
 *    filter : With shape [Co, 1, Hk, Wk].
 *    bias : With shape [Co], must be a Const-OP.
 *    offset_w : Reserved. For quantized.
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    strides : Stride along each axis.
 *    dilations : Dilation value along each axis of the filter.
 *    pads : Padding for the beginning and ending along each axis [hh, ht, wh, wt].
 *    pad_mode : Pad mode, 'SPECIFIC'(not set, only support in CPUCL); 'SAME', or 'VALID'.
 *    data_format : Format of operator, 'NCHW' or 'NHWC'. Default is 'NCHW'
 *    offset_x : Reserved. For quantized.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc filterTensorDesc(Shape({5, 1, 3, 4}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr filterTensor = std::make_shared<hiai::Tensor>(filterTensorDesc);
 *    vector<float> filterDataValue(5 * 1 * 3 * 4, 0);
 *    filterTensor->SetData((uint8_t*)filterDataValue.data(), 5 * 1 * 3 * 4 * sizeof(float));
 *    auto filter = hiai::op::Const("filter").set_attr_value(filterTensor);
 *
 *    TensorDesc biasTensorDesc(Shape({5}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr biasTensor = std::make_shared<hiai::Tensor>(biasTensorDesc);
 *    vector<float> biasDataValue(5, 0);
 *    biasTensor->SetData((uint8_t*)biasDataValue.data(), 5 * sizeof(float));
 *    auto bias = hiai::op::Const("bias").set_attr_value(biasTensor);
 *
 *    auto convolutionDepthwise = hiai::op::ConvolutionDepthwise("ConvolutionDepthwise")
 *                                .set_input_x(x)
 *                                .set_input_filter(filter)
 *                                .set_input_bias(bias)
 *                                .set_attr_strides({1, 1})
 *                                .set_attr_dilations({1, 1})
 *                                .set_attr_pads({0, 0, 0, 1})
 *                                .set_attr_pad_mode("SAME")
 *                                .set_attr_data_format("NCHW");
 */
HIAI_REG_OP(ConvolutionDepthwise)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_UINT8 }))
.HIAI_INPUT(filter, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_OPTIONAL_INPUT(bias, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OPTIONAL_INPUT(offset_w, TensorType({ DT_INT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_REQUIRED_ATTR(strides, AttrValue::LIST_INT)
.HIAI_ATTR(dilations, AttrValue::LIST_INT ({ 1, 1 }))
.HIAI_ATTR(pads, AttrValue::LIST_INT ({ 0, 0, 0, 0 }))
.HIAI_ATTR(pad_mode, AttrValue::STR { "SAME" })
.HIAI_ATTR(data_format, AttrValue::STR { "NCHW" })
.HIAI_ATTR(offset_x, AttrValue::INT { 0 })
.HIAI_OP_END()

/*
 * Quantized for ConvolutionDepthwise, computes a depthwise convolution from given input and filter tensors.
 * <Input>
 *    x : Input tensor with size [N, Ci, Hi, Wi]
 *    filter : With shape [Co, 1, Hk, Wk], must be a Const-OP.
 *             If filter_quant_type = 1, shall use type 'tensor(int8)'.
 *             for(i in C) { quant_filter[i] = origin_filter[i] / filter_quant_scales[i];}
 *    bias : With shape [Co], must be a const OP.
 *           If filter_quant_type = 1, shall use type 'tensor(int32)'.
 *           for(i in C) { quant_bias[i] = origin_bias[i] / (x_quant_scale * filter_quant_scales[i]);}
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    strides : Stride along each axis.
 *    dilations : Dilation value along each axis of the filter.
 *    pads : Padding for the beginning and ending along each axis [hh, ht, wh, wt].
 *    pad_mode : Pad mode, using pads; SAME, or VALID
 *    data_format : Only support NCHW. format of input, 'NCHW' or 'NHWC'. Default is 'NCHW'
 *    x_quant_type : Either 0-No quant or 1-Quant8 (linear); Reserved: 2-Quant4 (linear), 3-Quant2 (linear)
 *    filter_quant_type : Either 0-No quant or 1-Quant8 (linear); Reserved: 2-Quant4 (linear), 3-Quant2 (linear)
 *    x_quant_scale : If x_quant_type > 0, this Attr is required.
 *    x_quant_offset : Data offset
 *    filter_quant_scales : If filter_quant_type > 0, this attr is required.
 *                          must be 1 or n of [n, c, h, w] of 'filter'.
 * <Added in HiAI version>
 *    100.310.010.015
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc filterTensorDesc(Shape({5, 1, 3, 4}), FORMAT_NCHW, DT_INT8);
 *    TensorPtr filterTensor = std::make_shared<hiai::Tensor>(filterTensorDesc);
 *    vector<int8_t> filterDataValue(5 * 1 * 3 * 4, 0);
 *    filterTensor->SetData((uint8_t*)filterDataValue.data(), 5 * 1 * 3 * 4 * sizeof(int8_t));
 *    auto filter = hiai::op::Const("filter").set_attr_value(filterTensor);
 *
 *    TensorDesc biasTensorDesc(Shape({5}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr biasTensor = std::make_shared<hiai::Tensor>(biasTensorDesc);
 *    vector<int32_t> biasDataValue(5, 0);
 *    biasTensor->SetData((uint8_t*)biasDataValue.data(), 5 * sizeof(int32_t));
 *    auto bias = hiai::op::Const("bias").set_attr_value(biasTensor);
 *
 *    std::vector<ge::AttrValue::FLOAT> quantScales(5, 1.0);
 *    auto quantizedConvolutionDepthwise = hiai::op::QuantizedConvolutionDepthwise("quantizedConvolutionDepthwise")
 *                                         .set_input_x(x)
 *                                         .set_input_filter(filter)
 *                                         .set_input_bias(bias)
 *                                         .set_attr_strides({1, 1})
 *                                         .set_attr_dilations({1, 1})
 *                                         .set_attr_pads({0, 0, 0, 1})
 *                                         .set_attr_pad_mode("SAME")
 *                                         .set_attr_data_format("NCHW")
 *                                         .set_attr_x_quant_type(1)
 *                                         .set_attr_filter_quant_type(1)
 *                                         .set_attr_x_quant_scale(1.0)
 *                                         .set_attr_x_quant_offset(1.0)
 *                                         .set_attr_filter_quant_scales(quantScales);
 */
HIAI_REG_OP(QuantizedConvolutionDepthwise)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(filter, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_OPTIONAL_INPUT(bias, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(strides, AttrValue::LIST_INT)
.HIAI_ATTR(dilations, AttrValue::LIST_INT ({ 1, 1 }))
.HIAI_ATTR(pads, AttrValue::LIST_INT ({ 0, 0, 0, 0 }))
.HIAI_ATTR(pad_mode, AttrValue::STR { "SAME" })
.HIAI_ATTR(data_format, AttrValue::STR { "NCHW" })
.HIAI_ATTR(x_quant_type, AttrValue::INT { 0 })
.HIAI_ATTR(filter_quant_type, AttrValue::INT { 0 })
.HIAI_ATTR(x_quant_scale, AttrValue::FLOAT { 1.0 })
.HIAI_ATTR(x_quant_offset, AttrValue::INT { 0 })
.HIAI_ATTR(filter_quant_scales, AttrValue::LIST_FLOAT ({}))
.HIAI_OP_END()

/*
 * Computes an inner product with an input tensor, a set of learned weights and adds biases.
 * <Input>
 *    x : Input tensor
 *    w : Weight tensor.
 *    b : 1D tensor for bias.
 *    offset_w : Reserved. For quantized.
 * <Output>
 *    y : Output tensor.
 * <Attr>
 *    num_output : Number of neurons output after full connection.
 *    transpose : Reserved. Whether the weight matrix is transposed.
 *    axis : Reserved. Inner product calculation on the axis.
 *    offset_x : Reserved. For quantized.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc filterTensorDesc(Shape({4, 210, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr filterTensor = std::make_shared<hiai::Tensor>(filterTensorDesc);
 *    vector<float> filterDataValue(4 * 210 * 1 * 1, 0);
 *    filterTensor->SetData((uint8_t*)filterDataValue.data(), 4 * 210 * 1 * 1 * sizeof(float));
 *    auto filter = hiai::op::Const("filter").set_attr_value(filterTensor);
 *
 *    auto fullyConnection = hiai::op::FullyConnection("fullyConnection")
 *                         .set_input_x(x)
 *                         .set_input_w(filter)
 *                         .set_attr_num_output(4);
 */
HIAI_REG_OP(FullyConnection)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_UINT8 }))
.HIAI_INPUT(w, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_OPTIONAL_INPUT(b, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OPTIONAL_INPUT(offset_w, TensorType({ DT_INT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_REQUIRED_ATTR(num_output, AttrValue::INT)
.HIAI_ATTR(transpose, AttrValue::BOOL { false })
.HIAI_ATTR(axis, AttrValue::INT { 1 })
.HIAI_ATTR(offset_x, AttrValue::INT { 0 })
.HIAI_OP_END()

/*
 * Quantized for FullyConnection,
 * computes an inner product with an input tensor, a set of learned weights and adds biases.
 * <Input>
 *    x : Input tensor
 *    w : Weight tensor, must be a Const-OP.
 *        If w_quant_type = 1, shall use type 'tensor(int8)'.
 *        for(i in C) { quant_filter[i] = origin_filter[i] / filter_quant_scales[i];}
 *    b : 1D tensor for bias, must be a Const-OP.
 *        If w_quant_type = 1, shall use type 'tensor(int32)'.
 *        for(i in C) { quant_bias[i] = origin_bias[i] / (x_quant_scale * filter_quant_scales[i]);}
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    num_output : Number of neurons output after full connection.
 *    transpose : Reserved. Whether the weight matrix is transposed.
 *    axis : Reserved. Inner product calculation on the axis.
 *    x_quant_type : Either 0-No quant or 1-Quant8 (linear); Reserved: 2-Quant4 (linear), 3-Quant2 (linear)
 *    w_quant_type : Either 0-No quant or 1-Quant8 (linear); Reserved: 2-Quant4 (linear), 3-Quant2 (linear)
 *    x_quant_scale : If x_quant_type > 0, this Attr is required.
 *    x_quant_offset : Data offset
 *    w_quant_scales : If w_quant_type > 0, this Attr is required.
 *                     must be 1 or n of [n, c, h, w] of 'w'.
 * <Added in HiAI version>
 *    100.310.010.015
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc filterTensorDesc(Shape({4, 210, 1, 1}), FORMAT_NCHW, DT_INT8);
 *    TensorPtr filterTensor = std::make_shared<hiai::Tensor>(filterTensorDesc);
 *    vector<int8_t> filterDataValue(4 * 210 * 1 * 1, 0);
 *    filterTensor->SetData((uint8_t*)filterDataValue.data(), 4 * 210 * 1 * 1 * sizeof(int8_t));
 *    auto filter = hiai::op::Const("filter").set_attr_value(filterTensor);
 *
 *    auto quantizedFullyConnection = hiai::op::QuantizedFullyConnection("quantizedFullyConnection")
 *                                   .set_input_x(x)
 *                                   .set_input_w(filter)
 *                                   .set_attr_num_output(4)
 *                                   .set_attr_x_quant_type(1)
 *                                   .set_attr_w_quant_type(1)
 *                                   .set_attr_x_quant_scale(1.0)
 *                                   .set_attr_x_quant_offset(1.0)
 *                                   .set_attr_w_quant_scales({1.0});
 */
HIAI_REG_OP(QuantizedFullyConnection)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(w, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_OPTIONAL_INPUT(b, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(num_output, AttrValue::INT)
.HIAI_ATTR(transpose, AttrValue::BOOL { false })
.HIAI_ATTR(axis, AttrValue::INT { 1 })
.HIAI_ATTR(x_quant_type, AttrValue::INT { 0 })
.HIAI_ATTR(w_quant_type, AttrValue::INT { 0 })
.HIAI_ATTR(x_quant_scale, AttrValue::FLOAT { 1.0 })
.HIAI_ATTR(x_quant_offset, AttrValue::INT { 0 })
.HIAI_ATTR(w_quant_scales, AttrValue::LIST_FLOAT ({}))
.HIAI_OP_END()

/*
 * Pools the input tensors by taking the max, average, etc. within regions.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    mode : Either 0 max pooling), 1 (avg pooling), or 2 (L2 pooling)
 *    pad_mode : Pad mode, either 0 (NOTSET), 5 (VALID) or 6 (SAME). Defaults to 0 (NOTSET).
 *    global_pooling : Defaults to false. When global_pooling is true, window values are ignored.
 *    window : Window size, specifying the height and width of each filter. Here the size must be 2 or 1
 *             and value >= 1.
 *    pad : Pad size, specifying the number of pixels to (implicitly) add to each side of the input.
 *          Here the size must be 4 or 2 and value >= 0.
 *    stride : Stride size, specifying the intervals at which to apply the filters to the input.
 *             Here the size must be 2 or 1 and value >= 1.
 *    ceil_mode : Default 0 (floor: The largest integer not greater than the argument),
 *                or 1 (ceil: The smallest integer greater than argument)
 *    data_mode : Data mode, either 0 (rounded up) or 1 (rounded down)
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto poolingD = hiai::op::PoolingD("poolingD")
 *                    .set_input_x(x)
 *                    .set_attr_mode(0)
 *                    .set_attr_pad_mode(0)
 *                    .set_attr_global_pooling(false)
 *                    .set_attr_window({3, 3})
 *                    .set_attr_pad({0, 0, 0, 0})
 *                    .set_attr_stride({9, 9})
 *                    .set_attr_ceil_mode(1)
 *                    .set_attr_data_mode(0);
 */
HIAI_REG_OP(PoolingD)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_UINT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_UINT8 }))
.HIAI_ATTR(mode, AttrValue::INT { 0 })
.HIAI_ATTR(pad_mode, AttrValue::INT { 0 })
.HIAI_ATTR(global_pooling, AttrValue::BOOL { false })
.HIAI_ATTR(window, AttrValue::LIST_INT ({ 1, 1 }))
.HIAI_ATTR(pad, AttrValue::LIST_INT ({ 0, 0, 0, 0 }))
.HIAI_ATTR(stride, AttrValue::LIST_INT ({ 1, 1 }))
.HIAI_ATTR(ceil_mode, AttrValue::INT { 0 })
.HIAI_ATTR(data_mode, AttrValue::INT { 1 })
.HIAI_OP_END()

/*
 * Computes the element-wise product of input tensors, with the shape of the scale 'broadcast' to match
 * the shape of 'x': y = x * scale + bias.
 * <Input>
 *    x : Input tensor
 *    scale : Scale of the input.
 *    bias : Bias to add to the input.
 * <Output>
 *    y : Output tensor, with the same shape as 'x'
 * <Attr>
 *    axis : Reserved. Axis of input tensor along which to apply other tensors (scale or bias).
 *    num_axes : Reserved. First input is the number of covered axes.
 *    scale_from_blob : Reserved.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc filterTensorDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr filterTensor = std::make_shared<hiai::Tensor>(filterTensorDesc);
 *    vector<float> filterDataValue(4 * 5 * 6 * 7, 1.0);
 *    filterTensor->SetData((uint8_t*)filterDataValue.data(), 4 * 5 * 6 * 7 * sizeof(float));
 *    auto filter = hiai::op::Const("filter").set_attr_value(filterTensor);
 *
 *    TensorDesc biasTensorDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr biasTensor = std::make_shared<hiai::Tensor>(biasTensorDesc);
 *    vector<float> biasDataValue(4 * 5 * 6 * 7, 1.0);
 *    biasTensor->SetData((uint8_t*)biasDataValue.data(), 4 * 5 * 6 * 7 * sizeof(float));
 *    auto bias = hiai::op::Const("bias").set_attr_value(biasTensor);
 *
 *    auto scale = hiai::op::Scale("scale")
 *                 .set_input_x(x)
 *                 .set_input_scale(filter)
 *                 .set_input_bias(bias);
 */
HIAI_REG_OP(Scale)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(scale, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_INPUT(bias, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(axis, AttrValue::INT { 1 })
.HIAI_ATTR(num_axes, AttrValue::INT { 1 })
.HIAI_ATTR(scale_from_blob, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Randomly shuffles channels followed by a group convolutional layer, usually utilized for ShuffleNet CNN
 * architecture.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Tensor of same shape and type as 'x', shuffled according to 'group'
 * <Attr>
 *    group : Number of groups that input channels and output channels are divided into. Must be positive.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto shuffleChannel = hiai::op::ShuffleChannel("ShuffleChannel")
 *                         .set_input_x(x)
 *                         .set_attr_group(1);
 */
HIAI_REG_OP(ShuffleChannel)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_ATTR(group, AttrValue::INT { 1 })
.HIAI_OP_END()

/*
 * Shuffle the channels of the input tensor.
 * Given an input tensor and a integer value of num_groups, ShuffleChannelV2
 * divide the channel dimension into group, and reorganize the
 * channels by grouping channels with the same index in each group.
 * Along the channel dimension, the output is calculated using this formula:
 * output_channel[k * group + g] = input_channel[g * group_size + k]
 * where group_size = num_channels / group
 * The number of channels must be divisible by group.
 * Supported tensor rank: up to 4
 * <Input>
 *    x : An n-D tensor, specifying the tensor to be shuffled.
 * <Output>
 *    y : A tensor of same shape as x.
 * <Attr>
 *    axis : scalar, specifying the dimension
 *           channel shuffle would be performed on. Negative index is used to
 *           specify axis from the end (e.g. -1 for the last axis). Must be in
 *           the range [-n, n).
 *    group : scalar, specifying the number of groups.
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto shuffleChannelV2 = hiai::op::ShuffleChannelV2("ShuffleChannelV2")
 *                            .set_input_x(x)
 *                            .set_attr_group(1)
 *                            .set_attr_axis(0);
 */
HIAI_REG_OP(ShuffleChannelV2)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(axis, AttrValue::INT { 0 })
.HIAI_ATTR(group, AttrValue::INT { 1 })
.HIAI_OP_END()

/*
 * Computes the softmax (normalized exponential) values for given input.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output values, with the same shape as the x
 * <Attr>
 *    axis : Dimension softmax would be performed on
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto softmax = hiai::op::Softmax("softmax")
 *                   .set_input_x(x)
 *                   .set_attr_axis(1);
 */
HIAI_REG_OP(Softmax)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(axis, AttrValue::INT { 0 })
.HIAI_OP_END()

/*
 * Calculate values and indices of the k largest entries of the last dimension and output.
 * <Input>
 *    x : 1-D or higher Tensor with last dimension at least k
 *    k : Const OP. Number of top elements to look for along the last dimension
 * <Output>
 *    values : k largest elements along each last dimensional slice
 *    indices : indices of values within the last dimension of input
 * <Attr>
 *    sorted : If true, the resulting k elements will be sorted by the values in descending order.
 *             If it be set to false, just make sure the k elements no losing, the data set is whole.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(TopK)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8 }))
.HIAI_INPUT(k, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(values, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8 }))
.HIAI_OUTPUT(indices, TensorType({ DT_INT32 }))
.HIAI_ATTR(sorted, AttrValue::BOOL { true })
.HIAI_OP_END()

/*
 * Computes the LogSoftmax (normalized exponential) values for each layer in the batch of the given input.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output values, with the same shape as the input tensor
 * <Attr>
 *    axis : Dimension LogSoftmax would be performed on
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(LogSoftmax)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(axis, AttrValue::INT { -1 })
.HIAI_OP_END()

/*
 * Returns the rank of a tensor
 * <Input>
 *    x : Tensor of type float, int32, bool, uint8.
 * <Output>
 *    y : Rank of the input tensor.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(Rank)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_BOOL, DT_UINT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_INT32 }))
.HIAI_OP_END()

/*
 * Scatter 'x' into a new tensor according to 'indices'.
 * <Input>
 *    indices : An Index tensor.
 *    x : A tensor.
 *    shape : A 1-D const tensor.
 * <Output>
 *    y : Result, has same element type as x.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(ScatterNd)
.HIAI_INPUT(indices, TensorType({ DT_INT32 }))
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(shape, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Returns the XOR value of 'x1' and 'x2' element-wise.
 * <Input>
 *    x1 : First input operand
 *    x2 : Second input operand
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(LogicalXor)
.HIAI_INPUT(x1, TensorType({ DT_BOOL }))
.HIAI_INPUT(x2, TensorType({ DT_BOOL }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL }))
.HIAI_OP_END()

/*
 * This operator is equivalent to the activation function.
 * If the value is greater than threshold, the output is 1. Otherwise, the output is 0.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    threshold : Threshold value
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(Threshold)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(threshold, AttrValue::FLOAT { 0 })
.HIAI_OP_END()

/*
 * Transform axis-aligned bounding box proposals using bounding box deltas.
 * <Input>
 *    roi : Input tensor, the location of the bounding box proposals, the shape should be [num_rois, 4]
 *    bbox_deltas : Input tensor, the bounding box delta for each region of interest and each class,
 *                  the shape should be [num_rois, num_classes * 4]
 *    batch_split : Input tensor, the batch index of each box, the shape should be [num_rois]
 *    im_info : Input tensor, the information of each image in the batch, the shape should be [batches, 2]
 * <Output>
 *    y : Output tensor, the coordinates of each output bounding box for each class
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(AxisAlignedBboxTransform)
.HIAI_INPUT(roi, TensorType({ DT_FLOAT }))
.HIAI_INPUT(bbox_deltas, TensorType({ DT_FLOAT }))
.HIAI_INPUT(batch_split, TensorType({ DT_INT32 }))
.HIAI_INPUT(im_info, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * First of all normalize the elements, then scale the normalized data and get the output.
 * <Input>
 *    x1 : The input tensor to be normalized.
 *    x2 : The scaling parameters, a const op.The input x2 must be [x2N, x2C, 1, 1],
 *         if channel_shared is true, x2C must be 1,
 *         otherwise x2C must be the same with x1C.
 * <Output>
 *    y : The output tensor.
 * <Attr>
 *    across_spatial : Indicates whether standardization should cross spatial locations.
 *    channel_shared : The parameters used to control x2 are shared by multiple channels.
 *    eps : A very small float number used to avoid dividing by zero.
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x1 = hiai::op::Data("x1");
 *    x1.update_input_desc_x(xDesc);
 *
 *    TensorDesc xTensorDesc(Shape({3, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr xTensor = std::make_shared<hiai::Tensor>(xTensorDesc);
 *    vector<float> dataValue(3, 0.0);
 *    xTensor->SetData((uint8_t*)dataValue.data(), 3 * sizeof(float));
 *    auto x2 = hiai::op::Const("x2").set_attr_value(xTensor);
 *
 *    auto normalize = hiai::op::Normalize("normalize")
 *                     .set_input_x1(x1)
 *                     .set_input_x2(x2)
 *                     .set_attr_across_spatial(false)
 *                     .set_attr_channel_shared(true)
 *                     .set_attr_eps(2e-07f);
 */
HIAI_REG_OP(Normalize)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_ATTR(across_spatial, AttrValue::BOOL { false })
.HIAI_ATTR(channel_shared, AttrValue::BOOL { true })
.HIAI_ATTR(eps, AttrValue::FLOAT { 2e-07f })
.HIAI_OP_END()

/*
 * a kind of stateful layer derived from the notion that a densely connected layer that's processing
 * a sequence of input frames can be approximated by using a singular value decomposition of each of its nodes.
 * <Input>
 *    x : the input 2-D tensor.
 *    weights_feature : the weights_feature 2-D tensor.Only support Const op.
 *    weights_time : the weights_time 2-D tensor.Only support Const op.
 *    bias : the bias 1-D tensor.Only support Const op.
 *    state_in : the state(in) 2-D tensor.Only support Const op.
 * <Output>
 *    state_out : the state(out) tensor
 *    y : the output tensor
 * <Attr>
 *    rank : The rank of the SVD approximation.
 *    use_bias : true: use the bias tensor.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(SVDF)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(weights_feature, TensorType({ DT_FLOAT }))
.HIAI_INPUT(weights_time, TensorType({ DT_FLOAT }))
.HIAI_INPUT(bias, TensorType({ DT_FLOAT }))
.HIAI_INPUT(state_in, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(state_out, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(rank, AttrValue::INT)
.HIAI_ATTR(use_bias, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Performs parametric ReLU, produces one output data (Tensor)
 * where the function f(x) = slope * x for x < 0, f(x) = x for x >= 0
 * <Input>
 *    x : A multi-dimensional Tensor of type float32.
 *    weight : if data_format is 'NCHW', the shape of Slope tensor must be [1, C, 1, 1] or [C, 1, 1].
 *             if data_format is 'NHWC', the shape of Slope tensor must be [C].
 *             if x is 1-D or 2-D tensor, the shape of Slope tensor must be [C], C is the channel of x tensor.
 * <Output>
 *    y : Output values, with the same shape as the input tensor x
 * <Attr>
 *    data_format : format of input, 'NCHW' or 'NHWC'. Default is 'NCHW'
 * <Added in HiAI version>
 *    100.500.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc slopeTensorDesc(Shape({1, 5, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr slopeTensor = std::make_shared<hiai::Tensor>(slopeTensorDesc);
 *    vector<float> dataValue(5, 0.0);
 *    slopeTensor->SetData((uint8_t*)dataValue.data(), 5 * sizeof(float));
 *    auto slope = hiai::op::Const("slope").set_attr_value(slopeTensor);
 *
 *    auto prelu = hiai::op::PRelu("PRelu")
 *                 .set_input_x(x)
 *                 .set_input_weight(slope);
 */
HIAI_REG_OP(PRelu)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(weight, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(data_format, AttrValue::STR { "NCHW" })
.HIAI_OP_END()

/*
 * Calculates the swish function value of each element in the input tensor.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    scale : Scaling factor
 * <Added in HiAI version>
 *    100.600.020.100
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *    x.update_output_desc_y(xDesc);
 *    auto Swish = hiai::op::Swish("Swish")
 *                .set_input_x(x)
 *                .set_attr_scale(1);
 */
HIAI_REG_OP(Swish)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(scale, AttrValue::FLOAT { 1.0 })
.HIAI_OP_END()

/*
 * A Self Regularized Non-Monotonic Neural Activation Function
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.600.020.100
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *    x.update_output_desc_y(xDesc);
 *    auto Mish = hiai::op::Mish("Mish").set_input_x(x);
 */
HIAI_REG_OP(Mish)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes layer norm
 * <Input>
 *    x : Input tensor, a 2D-4D tensor
 *    gamma : A tensor, multiple to result
 *    beta : A tensor, add to result
 * <Output>
 *    y : Output tensor
 *    mean : Reserved
 *    variance : Reserved
 * <Attr>
 *    begin_norm_axis : Reserved.
 *    begin_params_axis : Reserved.
 *                        NOTE: By default, the preceding two parameters are set to 1 (standard LayerNorm).
 *                              Other values do not take effect.
 *    epsilon : A very small float number used to avoid dividing by zero.
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc gammaDesc(Shape({1, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data gamma = hiai::op::Data("gamma");
 *    gamma.update_input_desc_x(gammaDesc);
 *
 *    TensorDesc betaDesc(Shape({1, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data beta = hiai::op::Data("beta");
 *    beta.update_input_desc_x(betaDesc);
 *
 *    auto layerNorm = hiai::op::LayerNorm("layerNorm")
 *                     .set_input_x(x)
 *                     .set_input_gamma(gamma)
 *                     .set_input_beta(beta)
 *                     .set_attr_epsilon(1e-7f);
 */
HIAI_REG_OP(LayerNorm)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(gamma, TensorType({ DT_FLOAT }))
.HIAI_INPUT(beta, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_OUTPUT(mean, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_OUTPUT(variance, TensorType({ DT_FLOAT }))
.HIAI_ATTR(begin_norm_axis, AttrValue::INT { 1 })
.HIAI_ATTR(begin_params_axis, AttrValue::INT { 1 })
.HIAI_ATTR(epsilon, AttrValue::FLOAT { 1e-7f })
.HIAI_OP_END()

/*
 * Computes instance norm
 * <Input>
 *    x : Input tensor which supports 4D dimension format.
 *    gamma : A tesnor, multiple to result
 *    beta : A tensor, add to result
 * <Output>
 *    y : Output tensor
 *    mean : Reserved
 *    variance : Reserved
 * <Attr>
 *    data_format : format of input, 'NCHW' or 'NHWC'. Default is 'NCHW'
 *    epsilon : A very small float number used to avoid dividing by zero.
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc gammaDesc(Shape({1, 5, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data gamma = hiai::op::Data("gamma");
 *    gamma.update_input_desc_x(gammaDesc);
 *
 *    TensorDesc betaDesc(Shape({1, 5, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data beta = hiai::op::Data("beta");
 *    beta.update_input_desc_x(betaDesc);
 *
 *    auto instanceNorm = hiai::op::InstanceNorm("instanceNorm")
 *                     .set_input_x(x)
 *                     .set_input_gamma(gamma)
 *                     .set_input_beta(beta)
 *                     .set_attr_epsilon(1e-7);
 */
HIAI_REG_OP(InstanceNorm)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(gamma, TensorType({ DT_FLOAT }))
.HIAI_INPUT(beta, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_OUTPUT(mean, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_OUTPUT(variance, TensorType({ DT_FLOAT }))
.HIAI_ATTR(data_format, AttrValue::STR { "NCHW" })
.HIAI_ATTR(epsilon, AttrValue::FLOAT { 1e-7f })
.HIAI_OP_END()

/*
 * The priorbox operator is used to deploy the default box at each location in the feature map.
 * <Input>
 *    x : The input tensor.
 *    img : The information of image.
 * <Output>
 *    y : The output tensor
 * <Attr>
 *    min_size : Minimum value of box.
 *    max_size : Maximum value of box.
 *    aspect_ratio : Width-to-height ratio of boxes.
 *    flip : Flip each aspect ratio
 *    clip : Oversized boxes can be reduced to specified limits
 *    variance : Adjustment box variance
 *    step_h : The height of step.
 *    step_w : The width of step.
 *    offset : The offset of frame.
 *    img_h : The height of the image.
 *    img_w : The width of the image.
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({1, 16, 80, 80}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    img_h/img_w:
 *    std::vector<ge::AttrValue::FLOAT> variance =
 *        {0.10000000149011612, 0.10000000149011612, 0.20000000298023224, 0.20000000298023224};
 *    auto priorBox = hiai::op::PriorBox("priorBox")
 *                    .set_input_x(x)
 *                    .set_attr_min_size({30.0})
 *                    .set_attr_max_size({60.0})
 *                    .set_attr_aspect_ratio({1.0, 2.0})
 *                    .set_attr_flip(false)
 *                    .set_attr_variance(variance)
 *                    .set_attr_step_h(8.0)
 *                    .set_attr_step_w(8.0)
 *                    .set_attr_offset(0.5)
 *                    .set_attr_img_h(600)
 *                    .set_attr_img_w(600);
 *
 *    img:
 *    TensorDesc imgDesc(Shape({1, 3, 600, 600}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data img = hiai::op::Data("img");
 *    img.update_input_desc_x(imgDesc);
 *    std::vector<ge::AttrValue::FLOAT> variance =
 *        {0.10000000149011612, 0.10000000149011612, 0.20000000298023224, 0.20000000298023224};
 *    auto priorBox = hiai::op::PriorBox("priorBox")
 *                    .set_input_x(x)
 *                    .set_input_img(img)
 *                    .set_attr_min_size({30.0})
 *                    .set_attr_max_size({60.0})
 *                    .set_attr_aspect_ratio({1.0, 2.0})
 *                    .set_attr_flip(false)
 *                    .set_attr_variance(variance)
 *                    .set_attr_step_h(8.0)
 *                    .set_attr_step_w(8.0)
 *                    .set_attr_offset(0.5);
 */
HIAI_REG_OP(PriorBox)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_INPUT(img, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(min_size, AttrValue::LIST_FLOAT)
.HIAI_REQUIRED_ATTR(max_size, AttrValue::LIST_FLOAT)
.HIAI_REQUIRED_ATTR(aspect_ratio, AttrValue::LIST_FLOAT)
.HIAI_ATTR(flip, AttrValue::BOOL { true })
.HIAI_ATTR(clip, AttrValue::BOOL { false })
.HIAI_ATTR(variance, AttrValue::LIST_FLOAT ({ 0.1f }))
.HIAI_ATTR(step_h, AttrValue::FLOAT { 0 })
.HIAI_ATTR(step_w, AttrValue::FLOAT { 0 })
.HIAI_ATTR(offset, AttrValue::FLOAT { 0.5 })
.HIAI_ATTR(img_h, AttrValue::INT { 0 })
.HIAI_ATTR(img_w, AttrValue::INT { 0 })
.HIAI_OP_END()

/*
 * The function of the proposal operator is to obtain the optimal image region of the object
 * according to the migration of the default box and the probability of foreground.
 * <Input>
 *    cls_prob : Generate the foreground and background probabilities of the box.
 *    bbox_pred : Preselected box position.
 *    im_info : The image information.
 *    rpn_bbox : Number of Bboxs output by each batch.
 * <Output>
 *    rois : Output box information.
 *    actual_rois_num : Output box information.
 * <Attr>
 *    feat_stride : Specify the step size of two adjacent frames to extend H or W when generating the default box.
 *    base_size : The parameter used to generate the default box, indicating the basic size of the box.
 *    min_size : The minimum value of the side length of the box,
 *               all boxes smaller than this minimum will be filtered out.
 *    ratio : Generate the parameters used in the default box.
 *    scale : Generate the parameters used in the default box.
 *    pre_nms_topn : The number of points before the nms operation.
 *    post_nms_topn : The number of points after the nms operation.
 *    iou_thresh : Threshold used by the filter box must be (0, 1].
 *    output_actual_rois_num : Now support true.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(Proposal)
.HIAI_INPUT(cls_prob, TensorType({ DT_FLOAT }))
.HIAI_INPUT(bbox_pred, TensorType({ DT_FLOAT }))
.HIAI_INPUT(im_info, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_INPUT(rpn_bbox, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(rois, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(actual_rois_num, TensorType({ DT_FLOAT }))
.HIAI_ATTR(feat_stride, AttrValue::FLOAT { 16 })
.HIAI_ATTR(base_size, AttrValue::FLOAT { 16 })
.HIAI_ATTR(min_size, AttrValue::FLOAT { 16 })
.HIAI_ATTR(ratio, AttrValue::LIST_FLOAT ({ 0.5, 1, 2 }))
.HIAI_ATTR(scale, AttrValue::LIST_FLOAT ({ 32, 16, 8 }))
.HIAI_ATTR(pre_nms_topn, AttrValue::INT { 6000 })
.HIAI_ATTR(post_nms_topn, AttrValue::INT { 304 })
.HIAI_ATTR(iou_thresh, AttrValue::FLOAT { 0.7f })
.HIAI_ATTR(output_actual_rois_num, AttrValue::BOOL { true })
.HIAI_OP_END()

/*
 * Unidirectional Long Short-Term Memory layer, using an internal network unrolled in time.
 * <Input>
 *    x : The input tensor for LSTM cell. Shape is  [T, B, X], indicating max_time, batch_size and input.
 *    cont : Sequence continuation indicators. Shape is [T, B].
 *    w_x : Weight tensor, must be const op. Used between x and hidden units. Shape is [4 * hidden-size, X].
 *    bias : Bias for w_x. Shape is [4 * hidden-size].
 *    w_h : Weight tensor, must be const op. Used between time steps. Shape is [4 * hidden-size, hidden-size].
 *    x_static : Static data with constant time. Shape is [B, X].
 *    h_0 : Reserved. The input makes sense when the attr 'expose_hidden' is set to true.
 *    c_0 : Reserved. The input makes sense when the attr 'expose_hidden' is set to true.
 *    w_x_static : Weight tensor, must be const op. Used between xstatic and hidden units.
 *                 Shape is [4 * hidden-size, X].
 * <Output>
 *    h : The output tensor.
 *    h_t : The output tensor.
 *    c_t : The output tensor.
 * <Attr>
 *    expose_hidden : Reserved. Only support false now.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(LSTM)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(cont, TensorType({ DT_FLOAT }))
.HIAI_INPUT(w_x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(bias, TensorType({ DT_FLOAT }))
.HIAI_INPUT(w_h, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_INPUT(x_static, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_INPUT(h_0, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_INPUT(c_0, TensorType({ DT_FLOAT }))
.HIAI_OPTIONAL_INPUT(w_x_static, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(h, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(h_t, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(c_t, TensorType({ DT_FLOAT }))
.HIAI_ATTR(expose_hidden, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Pools the input tensors by taking the average, etc. within regions.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    ksize : ksize size, specifying the height and width of each filter. Here the size must be 2 or 1
 *            and value >= 1.
 *    strides : Stride size, specifying the intervals at which to apply the filters to the input.
 *              Here the size must be 2 or 1 and value >= 1.
 *    padding_mode : Pad mode, either NOTSET, VALID or SAME_UPPER. Defaults to NOTSET.
 *    pads : Pad size, specifying the number of pixels to (implicitly) add to each side of the input.
 *           Here the size must be 4 or 2 and value >= 0.
 *    data_format : Format of operator, 'NCHW' or 'NHWC'. Default is 'NCHW'
 *    global_pooling : Defaults to false. When global_pooling is true, ksize values are ignored.
 *    ceil_mode : Default false (floor: The largest integer not greater than the argument),
 *                or true (ceil: The smallest integer greater than argument)
 *    exclusive : Ignore padding area or not when calculating average.
 * <Added in HiAI version>
 *    100.600.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto avgPoolV2 = hiai::op::AvgPoolV2("avgPoolV2")
 *        .set_input_x(x)
 *        .set_attr_padding_mode("NOTSET")
 *        .set_attr_global_pooling(false)
 *        .set_attr_pads(AttrValue::LIST_INT({1, 1, 1, 1}))
 *        .set_attr_strides(AttrValue::LIST_INT({1, 1}))
 *        .set_attr_ksize(AttrValue::LIST_INT({3, 3}))
 *        .set_attr_ceil_mode(false)
 *        .set_attr_exclusive(true);
 */
HIAI_REG_OP(AvgPoolV2)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(ksize, AttrValue::LIST_INT)
.HIAI_REQUIRED_ATTR(strides, AttrValue::LIST_INT)
.HIAI_ATTR(padding_mode, AttrValue::STR { "NOTSET" })
.HIAI_ATTR(pads, AttrValue::LIST_INT ({ 0, 0, 0, 0 }))
.HIAI_ATTR(data_format, AttrValue::STR { "NCHW" })
.HIAI_ATTR(global_pooling, AttrValue::BOOL { false })
.HIAI_ATTR(ceil_mode, AttrValue::BOOL { false })
.HIAI_ATTR(exclusive, AttrValue::BOOL { true })
.HIAI_OP_END()

/*
 * Applies sparse updates to individual values or slices in a Variable.
 * <Input>
 *    var : A tensor. Tensor of rank r > 1"
 *    indices : An Index tensor. Tensor of rank r >= 1
 *    updates : A tensor. Tensor of rank is equal to (rank(data) + rank(indices) - indices.shape[-1] - 1)
 * <Output>
 *    var : Result, has same element type as data.
 * <Attr>
 *    use_locking : Reserved. Defaults to false. If true, the operation will be protected by a lock.
 * <Added in HiAI version>
 *    100.600.010.010
 */
HIAI_REG_OP(ScatterNdUpdate)
.HIAI_INPUT(var, TensorType({ DT_FLOAT }))
.HIAI_INPUT(indices, TensorType({ DT_INT32 }))
.HIAI_INPUT(updates, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(var, TensorType({ DT_FLOAT }))
.HIAI_ATTR(use_locking, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * computes root mean square normalization.
 * <Input>
 *    x : A ND tensor.
 *    gamma : A 1D tensor, multiple to result.
 * <Output>
 *    y : A ND tensor.
 *    rstd : Intermediate result, used for reverse calculation.
 * <Attr>
 *    epsilon : A very small float number used to avoid dividing by zero.
 * <Added in HiAI version>
 *    100.600.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc gammaDesc(Shape({7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data gamma = hiai::op::Data("gamma");
 *    gamma.update_input_desc_x(gammaDesc);
 *
 *    auto rmsNorm = hiai::op::RmsNorm("rmsNorm")
 *        .set_input_x(x)
 *        .set_input_gamma(gamma)
 *        .set_attr_epsilon(1e-6f);
 */
HIAI_REG_OP(RmsNorm)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
.HIAI_INPUT(gamma, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
.HIAI_OPTIONAL_OUTPUT(rstd, TensorType({ DT_FLOAT, DT_FLOAT, DT_FLOAT }))
.HIAI_ATTR(epsilon, AttrValue::FLOAT { 1e-6f })
.HIAI_OP_END()
} // namespace hiai
// clang-format on

#endif
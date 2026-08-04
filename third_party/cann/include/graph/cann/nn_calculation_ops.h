/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

/*!
 * \file nn_calculation_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_CALCULATION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Computes a 2D convolution given 4D "x", "filter" and "bias" tensors.
* Like this, output = CONV(x, filter) + bias.
* @par Inputs:
* @li x: A required 4D tensor of input image. With the format "NHWC" which shape is
* [n, h, w, in_channels] or the format "NCHW" which shape is [n, in_channels, h, w].
* @li filter: A required 4D tensor of convolution kernel.
* With the format "HWCN" which shape is [kernel_h, kernel_w, in_channels / groups, out_channels]
* or the format "NCHW" which shape is [out_channels, in_channels / groups, kernel_h, kernel_w].
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels].
* @li offset_w: An optional quantitative offset tensor. Reserved.
*\n
* The following are the supported data types and data formats (except IPV350 and Ascend 910_95 AI Processor):
*\n
| Tensor    | x        | filter   | bias     | y        |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | float16  | float16  | float16  | float32  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | bfloat16 | bfloat16 | bfloat16 | float32  |\n
|           | float32  | float32  | float32  | float32  |\n
|           | int8     | int8     | int32    | int32    |\n
| Format    | NCHW     | NCHW     | ND       | NCHW     |\n
|           | NHWC     | HWCN     | ND       | NHWC     |\n
|           | NCHW     | HWCN     | ND       | NCHW     |\n
*\n
* The following are the supported data types and data formats for IPV350:
*\n
| Tensor    | x       | filter  | bias    | y       |\n
| :-------: | :-----: | :-----: | :-----: | :-----: |\n
| Data Type | int16   | int8    | int32   | int32   |\n
|           | int8    | int8    | int32   | int32   |\n
| Format    | NCHW    | NCHW    | ND      | NCHW    |\n
|           | NHWC    | HWCN    | ND      | NHWC    |\n
*\n
* The following are the supported data types and data formats for Ascend 910_95 AI Processor:
*\n
| Tensor    | x        | filter   | bias     | y        |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | float32  | float32  | float32  | float32  |\n
|           | hifloat8 | hifloat8 | float32  | hifloat8 |\n
| Format    | NCHW     | NCHW     | ND       | NCHW     |\n
|           | NHWC     | HWCN     | ND       | NHWC     |\n
*\n
* @par Attributes:
* @li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NHWC", its shape is [1, stride_h, stride_w, 1],
* when the format is "NCHW", its shape is [1, 1, stride_h, stride_w].
* @li pads: Required. A list of 4 integers. The number of pixels to add to each
* (pad_top, pad_bottom, pad_left, pad_right) side of the input.
* @li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NHWC", its shape is [1, dilation_h, dilation_w, 1],
* when the format is "NCHW", its shape is [1, 1, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* @li data_format: Optional. It is a string represents input's data format.
* Defaults to "NHWC". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Ensure that the output is within the
* effective range. Defaults to 0. Reserved.
* @par Outputs:
* y: A 4D tensor of output feature map.
* With the format "NHWC" which shape is [n, out_height, out_width, out_channels]
* or the format "NCHW" which shape is [n, out_channels, out_height, out_width].
*\n
*     out_height = (h + pad_top + pad_bottom -
*                   (dilation_h * (kernel_h - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (w + pad_left + pad_right -
*                  (dilation_w * (kernel_w - 1) + 1))
*                 / stride_w + 1
*\n
* @attention Constraints:
* @li The following value range restrictions must be met:
*\n
| Name             | Field      | Scope       |\n
| :--------------: | :--------: | :---------: |\n
| x size           | h          | [1, 100000] |\n
|                  | w          | [1, 4096]   |\n
| filter size      | kernel_h   | [1, 255]    |\n
|                  | kernel_w   | [1, 255]    |\n
| strides          | stride_h   | [1, 63]     |\n
|                  | stride_w   | [1, 63]     |\n
| pads             | pad_top    | [0, 255]    |\n
|                  | pad_bottom | [0, 255]    |\n
|                  | pad_left   | [0, 255]    |\n
|                  | pad_right  | [0, 255]    |\n
| dilations        | dilation_h | [1, 255]    |\n
|                  | dilation_w | [1, 255]    |\n
| offset_x         | -          | [-128, 127] |\n
*\n
* @li The w dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
* @li If any dimension of x/filter/bias/offset_w/y shape exceeds max
* int32(2147483647), the product of each dimension of x/filter/bias/offset_w/y
* shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed. \n
* In Ascend 910_95 AI Processor: If any dimension of x/filter/bias/offset_w/y shape exceeds max
* 1000000, the product of each dimension of x/filter/bias/offset_w/y
* shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
*\n
* @li When the specifications of the Conv2D exceeds the constraints mentioned above,
* a timeout AI Core error may be reported.
*\n
* @par Quantization supported or not
* Yes
*\n
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator "conv2d".
* @li Compatible with the Caffe operator 2D "Convolution".
* @li Compatible with the ONNX operator 2D "Conv".
* @li Compatible with the PyTorch operator "Conv2D".
*/
REG_OP(Conv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2D)

/**
*@brief Computes the Deconvolution with respect to the input.
* @par Inputs:
 * Two required inputs:
 * @li x: A Tensor of type float16 or int8. 4D with shape
 * [batch, out_channels, out_height, out_width]. Gradients with respect
 * to the output of the convolution.
 * @li filter: A Tensor. Must have the same type as "x".
 * 4D with shape [out_channels, in_channel, filter_height, filter_width].
 \n
 * Two optional inputs:
 * @li bias: An optional tensor. Must have the same type as "y".
 * @li offset_w: An optional 1D tensor for quantized deconvolution.
 * Type is int8. Reserved.
 *\n
 *\n
 * The following are the supported data types and data formats:\n
 *\n
 *\n
    | Tensor    | x       | filter  | bias    | y      |\n
    |-----------|---------|---------|---------|--------|\n
    | Data Type | float16 | float16 | float16 | float16|\n
    |           | int8    | int8    | int32   | int32  |\n
    | Format    | NCHW    | NCHW    | ND      | NCHW   |\n
 *\n
 * For int8, a dequant or requant operator must be followed.
 *\n
 *
*@par Attributes:
 * Six attributes:
 * @li strides: A tuple or list of 2 integers. The stride of the sliding window
 * for H/W dimension, defaults to [1,1].
 * @li pads: A tuple or list of 4 integers. The [top, bottom, left, right]
 * padding on the feature map, defaults to [0,0,0,0].
 * @li dilations: A tuple or list of 4 integers. The dilation factor for each
 * dimension of input, defaults to [1,1,1,1].
 * @li groups: Number of blocked connections from input channels to
 * output channels. Defaults to "1".
 * @li data_format: An optional string from: "NCHW". Defaults to "NCHW". \n
 * Specify the data format of the input and output data.
 * @li offset_x: An optional integer for quantized deconvolution.
 * The negative offset added to the input image for int8 type. Ensure offset_x
 * within the effective range of int8 [-128, 127]. Defaults to "0".
 *\n
 *\n
 * The following value range restrictions must be met:\n
 *\n
 *\n
    | Name             | Field    | Scope        |\n
    |------------------|----------|--------------|\n
    | x (out_backprop) | H*strideH| [1, 4096]    |\n
    |                  | W*strideW| [1, 4096]    |\n
    | Filter           | H        | [1, 255]     |\n
    |                  | W        | [1, 255]     |\n
    | y (fmap)         | H        | [1, 4096]    |\n
    |                  | W        | [1, 4096]    |\n
    | Stride           | H        | [1, 63]      |\n
    |                  | W        | [1, 63]      |\n
    | Padding          | Top      | [0, 255]     |\n
    |                  | Bottom   | [0, 255]     |\n
    |                  | Left     | [0, 255]     |\n
    |                  | Right    | [0, 255]     |\n
    | Dilation         | H        | [1, 255]     |\n
    |                  | W        | [1, 255]     |\n
    | Offset_x         |          | [-128, 127]  |\n
 *\n
 * In Atlas Training Series Product, fmap or out_backprop's H and W not support 1 when\n
 * fmap_h + pad_top + pad_bottom != (filter_height - 1) * dilation_h + 1
 * and filter_width > fmap_width
 * If filter_h = 1 and filter_w = 1,
 *  out_backprop_w * stride_h * stride_w < 4096
 *\n
 *
*@par Outputs:
 * y: A Tensor. 4D tensor with shape [batch, channels, height, width].
 *\n
 *     out_backprop_height = (fmap_height + pad_top + pad_bottom -
 *                           (dilation_h * (filter_height - 1) + 1))
 *                           / stride_h + 1
 *\n
 *     out_backprop_width = (fmap_width + pad_left + pad_right -
 *                          (dilation_w * (filter_width - 1) + 1))
 *                          / stride_w + 1
 *\n
 *
 * When type of x is float16, the type of y must be float16.
 * When type of x is int8, the type of y must be int32.
*/
REG_OP(Deconvolution)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
    .ATTR(strides, ListInt, {1, 1})
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCHW")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Deconvolution)

/**
* @brief Computes a 2D depth-wise convolution given 4D "x", "filter" and "bias" tensors.
* Like this, output = DEPTHWISECONV(x, filter) + bias.
* @par Inputs:
* @li x: A required 4D tensor of input image. With the format "NHWC" which shape is
* [n, h, w, in_channels] or the format "NCHW" which shape is [n, in_channels, h, w].
* @li filter: A required 4D tensor of convolution kernel.
* With the format "HWCN" which shape is [kernel_h, kernel_w, 1, out_channels]
* or the format "NCHW" which shape is [out_channels, 1, kernel_h, kernel_w].
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels].
* @li offset_w: An optional quantitative offset tensor. Reserved.
*\n
* The following are the supported data types and data formats (except IPV350 and Ascend 910_95 AI Processor):
*\n
| Tensor    | x        | filter   | bias     | y        |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | float16  | float16  | float16  | float32  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | bfloat16 | bfloat16 | bfloat16 | float32  |\n
|           | float32  | float32  | float32  | float32  |\n
|           | int8     | int8     | int32    | int32    |\n
| Format    | NCHW     | NCHW     | ND       | NCHW     |\n
|           | NHWC     | HWCN     | ND       | NHWC     |\n
|           | NCHW     | HWCN     | ND       | NCHW     |\n
*\n
* The following are the supported data types and data formats for IPV350:
*\n
| Tensor    | x       | filter  | bias    | y       |\n
| :-------: | :-----: | :-----: | :-----: | :-----: |\n
| Data Type | int16   | int8    | int32   | int32   |\n
|           | int8    | int8    | int32   | int32   |\n
| Format    | NCHW    | NCHW    | ND      | NCHW    |\n
|           | NHWC    | HWCN    | ND      | NHWC    |\n
*\n
* The following are the supported data types and data formats for Ascend 910_95 AI Processor:
*\n
| Tensor    | x        | filter   | bias     | y        |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | float32  | float32  | float32  | float32  |\n
|           | hifloat8 | hifloat8 | float32  | hifloat8 |\n
| Format    | NCHW     | NCHW     | ND       | NCHW     |\n
|           | NHWC     | HWCN     | ND       | NHWC     |\n
*\n
* @par Attributes:
* @li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NHWC", its shape is [1, stride_h, stride_w, 1],
* when the format is "NCHW", its shape is [1, 1, stride_h, stride_w].
* @li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NHWC", its shape is [1, dilation_h, dilation_w, 1],
* when the format is "NCHW", its shape is [1, 1, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li pads: Required. A list of 4 integers. The number of pixels to add to each
* (pad_top, pad_bottom, pad_left, pad_right) side of the input.
* @li data_format: Optional. It is a string represents input's data format.
* Defaults to "NHWC". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Ensure that the output is within the
* effective range. Defaults to 0. Reserved.
* @par Outputs:
* y: A 4D tensor of output feature map.
* With the format "NHWC" which shape is [n, out_height, out_width, out_channels]
* or the format "NCHW" which shape is [n, out_channels, out_height, out_width].
*\n
*     out_height = (h + pad_top + pad_bottom -
*                   (dilation_h * (kernel_h - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (w + pad_left + pad_right -
*                  (dilation_w * (kernel_w - 1) + 1))
*                 / stride_w + 1
*\n
* @attention Constraints:
* @li The following value range restrictions must be met:
*\n
| Name             | Field      | Scope       |\n
| :--------------: | :--------: | :---------: |\n
| x size           | h          | [1, 100000] |\n
|                  | w          | [1, 4096]   |\n
| filter size      | kernel_h   | [1, 255]    |\n
|                  | kernel_w   | [1, 255]    |\n
| strides          | stride_h   | [1, 63]     |\n
|                  | stride_w   | [1, 63]     |\n
| pads             | pad_top    | [0, 255]    |\n
|                  | pad_bottom | [0, 255]    |\n
|                  | pad_left   | [0, 255]    |\n
|                  | pad_right  | [0, 255]    |\n
| dilations        | dilation_h | [1, 255]    |\n
|                  | dilation_w | [1, 255]    |\n
| offset_x         | -          | [-128, 127] |\n
*\n
* @li The w dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
* @li If any dimension of x/filter/bias/y shape exceeds max
* int32(2147483647), the product of each dimension of x/filter/bias/y
* shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed. \n
* In Ascend 910_95 AI Processor: If any dimension of x/filter/bias/y shape exceeds max
* 1000000, the product of each dimension of x/filter/bias/y
* shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
*\n
* @li When the specifications of the DepthwiseConv2D exceeds the constraints mentioned above,
* a timeout AI Core error may be reported.
*\n
* @par Quantization supported or not
* Yes
*\n
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator "depthwise_conv2d".
* @li Compatible with the Caffe operator "DepthwiseConvolution".
*/
REG_OP(DepthwiseConv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8, DT_FLOAT}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(DepthwiseConv2D)

/**
*@brief Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence . \n

*@par Inputs:
* @li x: A Tensor dtype of float16.
* @li cont: A Tensor dtype of float16, float32.
* @li w_x: A Tensor dtype of float16.
* @li bias: A Tensor dtype of int16, int32, float16, float32.
* @li w_h: A Tensor dtype of float16.
* @li x_static: A optinal Tensor dtype of float16.
* @li h_0: A optinal Tensor dtype of float16, float32.
* @li c_0: A optinal Tensor dtype of float16, float32.
* @li w_x_static: A optinal Tensor dtype of float16 . \n

*@par Attributes:
*@li num_output: A Scalar of output size dtype of int.
*@li expose_hidden: A Scalar(bool) of features hidden . \n

*@par Outputs:
*@li h: A Tensor dtype of float16, float32.
* @li h_t: A optinal Tensor dtype of float16, float32. The hidden state at time t.
* @li c_t: A optinal Tensor dtype of float16, float32. The cell state at time t . \n

*@par Third-party framework compatibility:
* Compatible with the Caffe operator LSTM.
*/
REG_OP(LSTM)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(cont, TensorType({DT_FLOAT32,DT_FLOAT16}))
    .INPUT(w_x, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16,DT_FLOAT32,DT_INT16,DT_INT32}))
    .INPUT(w_h, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(x_static, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(h_0, TensorType({DT_FLOAT16,DT_FLOAT32}))
    .OPTIONAL_INPUT(c_0, TensorType({DT_FLOAT16,DT_FLOAT32}))
    .OPTIONAL_INPUT(w_x_static, TensorType({DT_FLOAT16}))
    .OUTPUT(h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(h_t, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(c_t, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(num_output, Int, 0)
    .ATTR(expose_hidden, Bool, false)
    .OP_END_FACTORY_REG(LSTM)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_CALCULATION_OPS_H_

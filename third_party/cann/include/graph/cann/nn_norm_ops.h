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
 * \file nn_norm_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief Applies the Softmax function to an n-dimensional input tensor
*  rescaling them. so that the elements of the n-dimensional output tensor lie
*  in the range [0,1] and sum to 1.

* @par Inputs:
* One input:
* x: A mutable input tensor, which can be floating point tensors with different precisions. Must be one of the following data types: float16, float32, bfloat16,
* double. Should be a variable tensor. The format must be ND. Shape support 1D ~ 8D. \n

* @par Attributes:
* @li axes: An optional list of int. Specifies on which dimensions of input x the Softmax operation is performed.
* Multi-axis reduction is supported. Defaults to "{-1}".
* In Ascend 910_95 AI Processor, only single-axis reduction is supported. \n
* @li half_to_float: An optional bool. 
* This parameter determines whether to convert the output data type to float32 when the input data type is float16.
* Defaults to "false".
* - If true and the input data type is float16, the output data type should be float32.
* - Otherwise, the output data type should be the same as the input data type. \n

* @par Outputs:
* y: A ND tensor. The output tensor represents the probability distribution of the input tensor after being processed by the Softmax function.
* Has the same dimensionality and shape as the "x" with values in the range [0, 1].
* Must be one of the following types: float16, float32, bfloat16, double. \n

* @par Third-party framework compatibility
*  Compatible with the TensorFlow operator Softmax.
*/
REG_OP(SoftmaxV2)
    .INPUT(x, TensorType({ DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .OUTPUT(y, TensorType({ DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .ATTR(axes, ListInt, {-1})
    .ATTR(half_to_float, Bool, false)
    .OP_END_FACTORY_REG(SoftmaxV2)

/**
* @brief RmsNorm operator interface implementation. \n
*  calculating: x, gamma \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = gamma * (x * rstd)

* @par Inputs
* Two inputs, including:
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes
* epsilon: A optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Two outputs, including:
* @li y: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
*/
REG_OP(RmsNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(RmsNorm)

/**
* @brief LayernormV4 operator interface implementation \n
*  calculating: x, gamma, beta \n
*  mean  = np.mean(x, reduce_axis, keepdims=True) \n
*  rstd = np.rsqrt(np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = gamma*((x - mean) * rstd) + beta

*@par Inputs:
*Four inputs, including:
* @li x: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li normalized_shape: A ND Tensor. Must be one of the following types: int32, int64
* @li gamma: A ND Tensor. Must be one of the following types: float16, float32, bfloat16. Shape is normalized_shape.
* @li beta: A ND Tensor. Must be one of the following types: float16, float32, bfloat16. Shape is normalized_shape.\n

*@par Attributes:
* @li epsilon: An optional attribute, the type is float32. Defaults to 1e-5 . \n

*@par Outputs:
*Three outputs, including:
* @li y: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li mean: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li rstd: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
*/
REG_OP(LayerNormV4)
    .INPUT(x, "T1")
    .INPUT(normalized_shape, "T2")
    .OPTIONAL_INPUT(gamma, "T3")
    .OPTIONAL_INPUT(beta, "T4")
    .OUTPUT(y, "T5")
    .OUTPUT(mean, "T6")
    .OUTPUT(rstd, "T6")
    .ATTR(epsilon, Float, 0.00001f)
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32, DT_INT64}))
    .DATATYPE(T3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T4, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T5, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T6, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(LayerNormV4)

/**
*@brief Local Response Normalization .

*@par Inputs:
*One input, including:
*x: A Tensor. Must be 4-D shape, and only support the following types: float16, float32 . \n

*@par Attributes:
*@li depth_radius: An optional int32, specifying the half-width of the normalization window. Defaults to "5".
* under the caffe framework, if local_size is provided and is an odd number,
* depth_radius = (local_size - 1) / 2. local_size is the number of channels to sum over (for ACROSS_CHANNELS)
* or the side length of the square region to sum over (for WITHIN_CHANNEL).
*@li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
* Defaults to "1.0".
*@li alpha: An optional float32. A scaling factor, usually positive.
* Defaults to "1.0".
*@li beta: An optional float32. An exponent. Defaults to "0.75" for the caffe framework, Defaults to "0.5" for others.
*@li norm_region: An optional string. A mode option. "ACROSS_CHANNELS":0. Defaults to "ACROSS_CHANNELS" . \n

*@par Outputs:
*y: A Tensor. Has the same data type and shape as "x" . \n

* @attention Constraints:
* This operator will be deprecated in the future. Replace it with LayerNorm operator. \n

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator LRN.
*/
REG_OP(LRN)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(depth_radius, Int, 5)
    .ATTR(bias, Float, 1.0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.5)
    .ATTR(norm_region, String, "ACROSS_CHANNELS")
    .OP_END_FACTORY_REG(LRN)

/**
* @brief Scales the input .

* @par Inputs:
* Three inputs, including:
* @li x: An ND tensor of type float16 or float32 or bfloat16.
* @li scale: An ND tensor of type float16 or float32 or bfloat16
* @li bias: An optional ND tensor of type float16 or float32 or bfloat16. \n

* @par Attributes:
* @li axis: An optional int32 used to compute the shape of scale and bias input from the online bottoms.
        Defaults to "1".
* @li num_axes: An optional int32 used to compute the shape of scale and bias input from a Caffe model trained offline.
        Defaults to "1".
* @li scale_from_blob: An optional bool. If "true", scale and bias are input from a Caffe model trained offline.
        If "false", scale and bias are input from online bottoms. Defaults to "true" . \n

* @par Outputs:
* y: An ND tensor of type float16 or float32 or bfloat16. \n

* @attention Constraints:
* Assume that the shape length of "x" is "n" and that of "scale" is "m".
* @li "axis" is within the range [-n, n-1]. num_axes >= -1.
* @li If "scale_from_blob = true", "num_axes = -1", and "axis >= 0",
        the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < n-axis).
* If "axis < 0", the ith axis of "scale" and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < -axis).
* @li If "scale_from_blob = true" and "num_axes = 0", "scale" is a scalar with shape length 1 and dimension size 1.
* @li If "scale_from_blob = true", "num_axes > 0, and "axis >= 0", "axis + num_axes" must be less than or equal to "n"
        and the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
* If "axis < 0", "n + axis + num_axes" must be less than or equal to "n" and the ith axis of "scale"
        and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
* @li If "scale_from_blob = false", "scale" is not a scalar, and "axis >= 0","axis + m" must be less than or
        equal to "n" and the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < m).
* If "axis < 0", "n + axis + m" must be less than or equal to "n" and the ith axis of "scale" and
        the (i+n+"axis")th axis of "x" must have the same size (0 <= i < m).
* @li If "bias" is not None, the constraints for "bias" is the same as that for "scale".
* @par Third-party framework compatibility
* Compatible with the Caffe operator Scale.
*/
REG_OP(Scale)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16})) /* "First operand." */
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16})) /* "Second operand." */
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16})) /* "Third operand." */
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))  /* "Result, has same element type as x" */
    .ATTR(axis, Int, 1)
    .ATTR(num_axes, Int, 1)
    .ATTR(scale_from_blob, Bool, true)
    .OP_END_FACTORY_REG(Scale)

/**
*@brief Computes log softmax activations .

*@par Inputs:
*One input:
* logits: A ND tensor. Must be one of the following data types: double, bfloat16, float16, float32 . \n

*@par Attributes:
* axes: An optional list of ints. Multi-axis reduction is supported. Defaults to "{-1}" .
* In Ascend 910_95 AI Processor, only single-axis reduction is supported. \n

*@par Outputs:
* logsoftmax: A ND tensor. Has the same data type as "logits" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogSoftmax.
*/
REG_OP(LogSoftmaxV2)
    .INPUT(logits, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(logsoftmax, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxV2)

/**
*@brief Normalizes the input "x1" .

*@par Inputs:
* Two inputs, including:
*@li x1: A required NCHW or NHWC tensor of type float32, float16, or int8.
*@li x2: A required ND tensor of type float32, float16, or int8, specifying
* the scaling factor. If "channel_shared" is "true", "x2" is a [1]-dimensional
* vector. If "channel_shared" is "false", "x2" is a [C]-dimensional vector . \n

*@par Attributes:
*@li across_spatial: An optional bool, specifying the dimension of input "x1"
* to be summed. The value "true" (default) indicates dimensions C, H, W, and
* the value "false" indicates dimension C.
*@li channel_shared: An optional bool, specifying the dimension count of input
* "x2". The value "true" (default) indicates 1, and the value "false" indicates
* dimension C of "x1".
*@li eps: An optional float32, specifying the bias when "across_spatial" is
* "true". Defaults to "1e-10" . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as "x1" . \n

*@par Third-party framework compatibility
* Compatible with the Caffe operator Normalize.
*/
REG_OP(Normalize)
     .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
     .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
     .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
     .ATTR(across_spatial, Bool, true)
     .ATTR(channel_shared, Bool, true)
     .ATTR(eps, Float, 1e-10f)
     .OP_END_FACTORY_REG(Normalize);
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_

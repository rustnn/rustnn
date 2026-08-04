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
 * \file nn_pooling_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Upsample the layer, similar to the nearest-neighbor
* difference scaling algorithm.

* @par Inputs:
* one input, including:
* x: A tensor of type float16 or float32. Supported format "NC1HWC0".
* Shape support 5D.
* @par Attributes:
* @li  scale: A optional float32, scale factor of x. Defaults to "1".
* @li  stride_h: An optional int, broadcast the axis of h. Defaults to "2".
* @li  stride_w: An optional int, broadcast the axis of w. Defaults to "2".
* @par Outputs:
* y: A tensor of type float16 or float32. Has same dtype as "x".
* Supported format "NC1HWC0". Shape support 5D.
*/
REG_OP(Upsample)
   .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
   .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
   .ATTR(scale, Float, 1)
   .ATTR(stride_h, Int, 2)
   .ATTR(stride_w, Int, 2)
   .OP_END_FACTORY_REG(Upsample)

/**
* @brief Performs pooling on the input.

* @par Inputs:
* x: An NCHW tensor of type float16, float32, int8.
* @par Attributes:
* @li mode: An optional int32, specifying the pooling algorithm, either "0" (max 
* pooling) or "1" (avg pooling). Defaults to "0".
* @li global_pooling: An optional bool. Defaults to "false".
* @li window: Optional, including:
* window[0]: An optional int32, specifying the window size along in the H 
* dimension. The value range is [1, 32768]. Defaults to "1".
* window[1]: An optional int32, specifying the window size along in the W 
* dimension. The value range is [1, 32768]. Defaults to "1".
* @li stride: Optional, including:
* stride[0]: An optional int32, specifying the stride along in the H dimension.
* The value range is [1, 63]. Defaults to "1".
* stride[1]: An optional int32, specifying the stride along in the W dimension.
* The value range is [1, 63]. Defaults to "1".
* @li pad: Optional, including:
* pad[0]: An optional int32, specifying the up padding. Defaults to "0".
* pad[1]: An optional int32, specifying the bottom padding. Defaults to "0".
* pad[2]: An optional int32, specifying the left padding. Defaults to "0".
* pad[3]: An optional int32, specifying the right padding. Defaults to "0".
* @li dilation: Optional, including:
* dilation[0]: An optional int32, specifying the up dilation. Defaults to "1".
* dilation[1]: An optional int32, specifying the bottom dilation. Defaults to 
* "1".
* dilation[2]: An optional int32, specifying the left dilation. Defaults to "1".
* dilation[3]: An optional int32, specifying the right dilation. Defaults to
* "1".
* @li ceil_mode: An optional int32, either "0" (ceil mode) or "1" (floor mode).
* Defaults to "0".
* @li data_format: An optional string, Specify the data format of the input and
* output data. With the default format "NCHW".
* @par Outputs:
* y: An NCHW tensor of type float16, float32, int32. \n
* The shape relationship between y and x as follows: \n
* Ny = Nx \n
* Cy = Cx \n
* Hy = (ceil_mode(Hx + pad[0] + pad[1] - window[0]) / stride[0]) + 1 \n
* Wy = (ceil_mode(Wx + pad[2] + pad[3] - window[1]) / stride[1]) + 1 \n
* @attention Constraints:
* @li Type float32 is only for dynamic shape.
* @li window[0] * window[1] < 256.
* @li 1<=Hx<=4096,1<=Wx<=4096.
* @li If input tensor N is a prime number, it should be less than 65535.
* @par Third-party framework compatibility
* @li Compatible with the Caffe operator Pooling.
* @li Compatible with the TensorFlow operator Pooling.
*/
REG_OP(Pooling)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT32}))
    .ATTR(mode, Int, 0)                 // 0:max pooling or 1:avg pooling
    .ATTR(global_pooling, Bool, false)
    .ATTR(window, ListInt, {1,1})       // kernel size
    .ATTR(stride, ListInt, {1,1})       // stride size
    .ATTR(pad, ListInt, {0,0,0,0})      // pad size
    .ATTR(dilation, ListInt, {1,1,1,1})
    .ATTR(ceil_mode, Int, 0)
    .ATTR(data_format, String, "NCHW")
    .OP_END_FACTORY_REG(Pooling)

/**
* @brief Performs average pooling on the input.

* @par Inputs:
* x: A tensor of shape [N, C, H, W] or [N, H, W, C] which supports data type float16, float32, double.

* @par Attributes:
* @li ksize: A required ListInt, list of 4 ints, specifying the size (N, C, H, and W)
* of the sliding window, where N = C = 1,
 * and H and W are positive integers within the range [1, 255].
* @li strides: A required ListInt, list of 4 ints, specifying the stride of the
 * sliding window. The strides of the N and C dimensions are 1.
 * The strides of the H and W dimensions are positive integers within
 * the range [1, 63].
* @li padding_mode: An optional String, specifying the padding algorithm,
 * either "VALID", "SAME" and "CALCULATED".
 * With "SAME" means that the outputs will have the same spatial dimensions
 * as its inputs. With "VALID" means no padding.
* @li pads: A optional ListInt. Pad value when padding_mode is "CALCULATED".
* @li data_format: An optional String, specifying the data format of "ksize"
 * and "strides", either "NHWC", or "NCHW" (default).
* @li global_pooling: An optional Bool. Global or not. If true, pads will change to {0,0,0,0}
* and ksize will change to [input_h, input_w].
* @li ceil_mode: An optional Bool. Use ceil or floor to calculate the output size when
* padding_mode is "CALCULATED".
* @li exclusive: An optional Bool. Ignore padding area or not when calculating average.
* @li divisor_override: An optional Int, its valid range is [1, 255], and the default value is zero.
* if specified, it will be used as divisor, otherwise size of the pooling region will be used.

* @par Outputs:
* y: The average pooled output tensor. Has the same type and format as
* input "x".

* @attention Constraints:
* @li Only single input and single output are supported.
* @li Global pooling is supported.
* @li "ksize_H" and "ksize_W" are positive integers within the range [1, 255].
* ksize_H * ksize_W < 256
* @li Due to instruction restrictions,
 * the values of "strides_h" and "strides_w" are positive integers within
 * the range [1, 63].
* @li If the sliding window range exceeds the original width and height of the input feature map,
 * and the calculation result of count_include_pad is False, the behavior of dividing by 0 will appear.
 * This scenario does not conform to the normal logic of the operator.
 * It is recommended to modify attributes such as ceil_mode or stride to satisfy that the sliding window
 * always has an intersection with the input feature map. In this abnormal scenario,
 * different chips may return different results, and four abnormal results may appear: 0, 65504, Nan, and INF.
* @li When the C axis is greater than 1, if points with the same H and W dimensions in x contain one INF input
 * on the C axis, the output of the INF input covered by the sliding window on this C axis is INF, and the
 * outputs of other C axis without INF input covered by the sliding window are Nan. If points with the same
 * H and W dimensions in x contain more than one INF input on the C axis, the outputs of all INF input data
 * covered by the sliding window on the C axis are Nan.
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPoolV2.
*/
REG_OP(AvgPoolV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(exclusive, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .OP_END_FACTORY_REG(AvgPoolV2)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H

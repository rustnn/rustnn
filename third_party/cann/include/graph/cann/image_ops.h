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
 * \file image_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_IMAGE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_IMAGE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Resize images to size using bilinear interpolation . \n

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include:
*@li x: 4-D tensor. Must set the format, supported format list ["NCHW, NHWC"]
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new
size for the images . \n

*@par Attributes:
* @li align_corners: An optional bool. If true, the centers of the 4 corner pixels of the input and
output tensors are aligned, preserving the values at the corner pixels.
Defaults to false .
* @li half_pixel_centers: An optional bool. If true, the center of pixels locate in [0.5, 0.5]. 
Defaults to False . 
* @li dtype: An optional Type attr, support type list [uint8, float32, float16, bfloat16].
The data type of output y.
Defaults to float32 . 
* @li scales: An optional listfloat. Multiplier for spatial size. Defaults to {0.0f, 0.0f} . 
*@par Outputs:
*y: 4-D tensor, format must be the same as x. support format list ["NCHW", "NHWC"]. 
When the dtype of y is float32, the dtype of x can be float32, float16 or bfloat16. When the dtype of y is
float16 or bfloat16, then the dtype of y must be the same as x.

*@par Third-party framework compatibility
*Compatible with tensorflow and pytorch ResizeBilinearV2 operator.
*/

REG_OP(ResizeBilinearV2)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                          DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_UINT8, DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .ATTR(dtype, Type, DT_FLOAT)
    .ATTR(scales, ListFloat, {0.0f, 0.0f})
    .OP_END_FACTORY_REG(ResizeBilinearV2)

/**
* @brief Resize images to size using nearest neighbor interpolation. \n

* @par Inputs:
* Inputs include:
* @li x: A 4-D tensor. Represents the original image. Must set the format, supported format list ["NCHW, NHWC"].
* Must be one of the following types: int8, uint8, int16, uint16, int32, int64, float16, float32,
* double, bfloat16.
* @li size: A 1-D int32 tensor of 2 elements: new_height, new_width.
* Indicates the size of the target image, which is used to determine the height and width of the output image.
* Must be the type int32. \n

* @par Attributes:
* @li align_corners: An optional bool. Determines whether to align the corners of the input and output images.
* If set to True, the corner pixels of the input and output images are aligned,
* preserving the value of the corner pixels. When set to false,
* the scaling process scales according to proportions and does not strictly align the corners.
* Defaults to false.
* @li half_pixel_centers: An optional bool. Determines the pixel center position during interpolation.
* If this parameter is set to True, the interpolation algorithm considers the center point of the pixel
* to estimate the pixel value more accurately. When set to false, the pixel center is on the integer coordinate point.
* Defaults to false. \n

* @li scales: An optional listfloat. Multiplier for spatial size. Defaults to {0.0f, 0.0f} .
* @par Outputs:
* y: A 4-D tensor. Indicates the target image. Has the same type and format as input "x". \n

* @par Third-party framework compatibility
* Compatible with tensorflow ResizeNearestNeighbor operator.
*/

REG_OP(ResizeNearestNeighborV2)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                          DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                           DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .ATTR(scales, ListFloat, {0.0f, 0.0f})
    .OP_END_FACTORY_REG(ResizeNearestNeighborV2)

/**
*@brief Resize images to size using bicubic interpolation . \n

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include:
*@li images: 4-D with shape [batch, height, width, channels] (format is NHWC) or
[batch, channels, height, width] (format is NCHW).
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new
size for the images . \n

*@par Attributes:
*@li align_corners: An optional bool. If true, the centers of the 4 corner pixels of the input
and output tensors are aligned, preserving the values at the corner pixels.
Defaults to false.
*@li half_pixel_centers: An optional bool. Defaults to False .
*@li dtype: An optional Type attr. Determine the DataType of input tensor and output tensor,
must be float (set value 0) or uint8 (set value 4) , defaults to float.\n

*@par Outputs:
*y: 4-D with shape [batch, height, width, channels] (format is NHWC) or
[batch, channels, height, width] (format is NCHW). \n

*@attention Constraints:
*Input images can be of different types, output images must be float or uint8.

*@par Third-party framework compatibility
*Compatible with tensorflow ResizeBicubic operator.
*/

REG_OP(ResizeBicubic)
    .INPUT(images, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_UINT8, DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(ResizeBicubic)

/**
* @brief Resize the input tensor. \n
currently, only support resize image tensor using nearest neighbor and linear interpolation.

* @par Inputs:
* Input x must be a 4-D tensor. Inputs include: \n
* @li x: A Tensor. Must be one of the following types: uint8, int8, int16, \n
int32, int64, float16, float, double. 4-D with shape [batch, height, width, channels] \n
or shape [batch, channels, height, width].
* @li roi: A 1-D float Tensor. Only takes effect when attr coordinate_transformation_mode \n
is "tf_crop_and_resize". Must be one of the following types: float16, float, double.
* @li scales: A 1-D float Tensor, the scale array along each dimension, Only one of \n
'scales' and 'sizes' can be specified. Must be float type.
* @li sizes: A 1-D int64 Tensor, The size of the output tensor. Only one of \n
'scales' and 'sizes' can be specified.  If 'size' is specified, then set scales \n
to empty data (zero shape) in this operator's input list. Must be one of \n
the following types: int32, int64.

* @par Attributes:
* @li coordinate_transformation_mode: An optional String. how to transform \n
the coordinate in the resized tensor to the coordinate in the original tensor. \n
options: pytorch_half_pixel, align_corners, asymmetric, \n
tf_crop_and_resize.
* @li cubic_coeff_a: An optional Float. Defaults to -0.75, only used in cubic interpolation. \n
other optional: -0.5
* @li exclude_outside: An optional Int. Defaults to 0, If set to 1, the weight of sampling \n
locations outside the tensor will be set to 0 and the weight will be renormalized \n
so that their sum is 1.0.
* @li extrapolation_value: An optional Float. Defaults to 0.0f. When coordinate_transformation_mode \n
is "tf_crop_and_resize" and x_original is outside the range [0, length_original - 1], \n
this value is used as the corresponding output value.
* @li mode: An optional String. Defaults to nearest. Three interpolation modes: nearest (default), \n
linear and cubic.
* @li nearest_mode: An optional String. Defaults to round_prefer_floor. Four modes: round_prefer_floor, \n
round_prefer_ceil, floor, ceil. Only used by nearest interpolation.

* @par Outputs:
* y: A Tensor. Has the same type as x.

* @attention Constraints: \n
* Input x must be a 4-D tensor.

* @par Third-party framework compatibility
* Compatible with tensorflow ResizeNearestNeighborV2 operator.
*/

REG_OP(Resize)
    .INPUT(x, TensorType({DT_INT8,DT_UINT8,DT_INT16,DT_UINT16,DT_INT32,
                          DT_INT64,DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .OPTIONAL_INPUT(roi, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(sizes, TensorType({DT_INT64,DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8,DT_UINT8,DT_INT16,DT_UINT16,DT_INT32,
                           DT_INT64,DT_FLOAT16,DT_FLOAT,DT_DOUBLE}))
    .ATTR(coordinate_transformation_mode, String, "half_pixel")
    .ATTR(cubic_coeff_a, Float, -0.75)
    .ATTR(exclude_outside, Int, 0)
    .ATTR(extrapolation_value, Float, 0.0)
    .ATTR(mode, String, "nearest")
    .ATTR(nearest_mode, String, "round_prefer_floor")
    .OP_END_FACTORY_REG(Resize)

/**
* @brief Extracts crops from the input image tensor and resizes them. Extracts
crops from the input image tensor and resizes them using bilinear sampling or
nearest neighbor sampling to a common output size specified by crop_size . \n

* @par Inputs:
* Input x must be a 4-D tensor. Inputs include:
* @li x: A Tensor. Must be one of the following types:uint8, uint16, int8,
         int16, int32, int64, float16, float, double. A 4-D tensor of shape
         [batch, image_height, image_width, depth]. The format must be NHWC.
* @li boxes: A Tensor. Must be the float types.
             A 2-D tensor of shape [num_boxes, 4].
* @li box_index: A Tensor of type int32. A 1-D tensor of shape [num_boxes] with
                 int32 values in [0, batch).
* @li crop_size: A Tensor of type int32. A 1-D tensor of 2 elements,
                 crop_size = [crop_height, crop_width].
                 All cropped image patches are resized to this size . \n

* @par Attributes:
* @li extrapolation_value: An optional float. Defaults to 0. Value used for
                           extrapolation, when applicable.
* @li method: An optional string from: '"bilinear", "nearest"'. Defaults to
              "bilinear". Currently two sampling methods are supported: Bilinear
              and NearestNeighbor .
* @li dtype: An optional Type attr, support type list [uint8, float16, float].
             Defaults to DT_FLOAT . \n

* @par Outputs:
* y: A Tensor. Must be one of the following types: uint8, float16, float.
     The format must be NHWC. \n

* @attention Constraints:
* Input images must be a 4-D tensor . \n

* @par Third-party framework compatibility
* Compatible with tensorflow CropAndResize operator.
*/

REG_OP(CropAndResizeV2)
    .INPUT(x, TensorType({DT_UINT8, DT_UINT16, DT_INT8, \
        DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .INPUT(crop_size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT}))
    .ATTR(extrapolation_value, Float, 0)
    .ATTR(method, String, "bilinear")
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(CropAndResizeV2)

/**
* @brief Greedily selects a subset of bounding boxes in descending order of
* score . \n

* @par Inputs:
* Input boxes and  scores must be float type. Inputs include:
* @li boxes: A 2-D float tensor of shape [num_boxes, 4]. They are expected to be in (x1, y1, x2, y2)
* format with x1 < x2 and y1 < y2.
* @li scores: A 1-D float tensor of shape [num_boxes] representing a single
* score corresponding to each box (each row of boxes).
* @li max_output_size: A scalar integer tensor representing the maximum number
* of boxes to be selected by non max suppression.
* @li iou_threshold: A 0-D float tensor representing the threshold for deciding
* whether boxes overlap too much with respect to IOU.
* @li score_threshold: A 0-D float tensor representing the threshold for
* deciding when to remove boxes based on score . \n

* @par Attributes:
* offset: An optional int. Defaults to 0. \n

* @par Outputs:
* selected_indices: A 1-D integer tensor of shape [M] representing the selected
* indices from the boxes tensor, where M <= max_output_size . \n

* @attention Constraints:
* Input boxes and  scores must be float type . \n

* @par Third-party framework compatibility
* Compatible with tensorflow NonMaxSuppressionV3 operator.
*/

REG_OP(NonMaxSuppressionV3)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(max_output_size, TensorType({DT_INT32}))
    .INPUT(iou_threshold, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(score_threshold, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .ATTR(offset, Int, 0)
    .OP_END_FACTORY_REG(NonMaxSuppressionV3)

/**
*@brief This operation samples input x by using interpolation based on flow 
*field grid, which is usually gennerated by affine_grid. The grid of shape 
*[N, H, W, 2] is the concatenation of (x, y) coordinates with shape [N, H, W] 
*each, where x is indexing the 4th dimension (in width dimension) of input 
*data x and y is indexng the 3rd dimention (in height dimension), finally 
*results is the interpolation value of 4 nearest corner points. The output 
*tensor shape will be [N, C, H, W].

*@par Inputs:
*@li x: 4-D Tensor with shape `[batch, channels, height, width]`. Must be one 
*of the following types: float16, float, double.
*@li grid: flow field grid, 4-D Tensor with shape `[batch, height, width, 2]` 
*and has same dtype as `x`. \n

*@par Attributes:
*@li interpolation_mode: An optional string specifying the interpolation 
*method, either 'bilinear', 'nearest' and 'bicubic'. Defaults to 
*"bilinear".
*@li padding_mode: An optional string specifying the pad method, either 
*"zeros", "border", or "reflection". Defaults to "zeros".
*@li align_corners: An optional bool. If "true", the centers of the corner
*pixels of the input and output tensors are aligned. Defaults to "false" . \n

*@par Outputs:
*y: Returns 4-D Tensor with the same dtype as `x`. \n

*@par Third-party framework compatibility
*Compatible with pytorch GridSampler2D operator.
*/
REG_OP(GridSampler2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(grid, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(interpolation_mode, String, "bilinear")
    .ATTR(padding_mode, String, "zeros")
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(GridSampler2D)

/**
* @brief NormalizeV2 \n

* @par Inputs:
* @li x: A 4-D Tensor. Must be one of the following types: uint8, float16,
*        float. Must set the format, supported format list ["NCHW, NHWC"].
* @li mean: A 4-D float tensor. value of "C(channel)" is same to x
* @li variance: A 4-D float tensor. value of "C(channel)" is same to x \n

* @par Outputs:
* @li y: A 4-D Tensor. Must be one of the following types: float16, float.
*        Must set the format, supported format list ["NCHW, NHWC"]. \n

* @par Attributes:
* @li dtype: An Type attr, support type list [DT_FLOAT16, DT_FLOAT].
*            Defaults to DT_FLOAT. \n

* @par Third-party framework compatibility
* Compatible with pytorch normalize operator.
*/

REG_OP(NormalizeV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(NormalizeV2)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_IMAGE_OPS_H_

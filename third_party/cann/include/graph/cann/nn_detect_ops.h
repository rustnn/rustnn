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
 * \file nn_detect_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Returns detection result .

*@par Inputs:
* Three inputs, including:
*@li bbox_delta: An ND tensor of type floa16 or float32, specifying the box
*loc predictions, used as the input of operator SSDDetectionOutput.
*@li score: An ND tensor of type floa16 or float32, specifying the box
*confidences data, used as the input of operator SSDDetectionOutput.
*@li anchors: An ND tensor of type floa16 or float32, output from operator
*PriorBoxD, used as the input of operator SSDDetectionOutput. \n

*@par Attributes:
*@li num_classes: An optional int, specifying the number of classes to be
*predicted. Defaults to "2". The value must be greater than 1 and lesser
*than 1025.
*@li share_location: An optional bool, specify the shared location.
*Defaults to true.
*@li background_label_id: An optional int, specify the background label id.
*Must be 0.
*@li iou_threshold: An optional float, specify the nms threshold. Default to
*0.3.
*@li top_k: An optional int, specify the topk value. Defaults to 200.
*@li eta: An optional float, specify the eta value. Defaults to 1.0.
*@li variance_encoded_in_target: An optional bool, specify whether variance
*encoded in target or not. Defaults to false.
*@li code_type: An optional int, specify the code type. Defaults to 1. 
*The corner is 1, center_size is 2, corner_size is 3.
*@li keep_top_k: An optional int, specify the topk value after nms.
*Defaults to -1.
*@li confidence_threshold: An optional float, specify the topk filter threshold.
* Only consider detections with confidence greater than the threshold. \n

*@par Outputs:
*@li out_boxnum: A tensor of type int32, specifying the number of output boxes.
*@li y: A tensor of type float16 or float32 with shape [batch,keep_top_k, 8],
*describing the information of each output box.
* In output shape, 8 means (batchID, label(classID), score (class probability),
*xmin, ymin, xmax, ymax, null).
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(SSDDetectionOutput)
    .INPUT(bbox_delta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(score, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(anchors, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(out_boxnum, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(num_classes, Int, 2)
    .ATTR(share_location, Bool, true)
    .ATTR(background_label_id, Int, 0)
    .ATTR(iou_threshold, Float, 0.3f)
    .ATTR(top_k, Int, 200)
    .ATTR(eta, Float, 1.0)
    .ATTR(variance_encoded_in_target, Bool, false)
    .ATTR(code_type, Int, 1)
    .ATTR(keep_top_k, Int, -1)
    .ATTR(confidence_threshold, Float, 0.0)
    .OP_END_FACTORY_REG(SSDDetectionOutput)

/**
* @brief Obtains the ROI feature matrix from the feature map. It is a customized FasterRcnn operator . \n

* @par Inputs:
* Three inputs, including:
* @li features: A 5HD Tensor of type float32 or float16.
* @li rois: ROI position. A 2D Tensor of float32 or float16 with shape (N, 5). "N" indicates the number of ROIs,
* the value "5" indicates the indexes of images where the ROIs are located,
* "x0", "y0", "x1", and "y1".
* @li rois_n: An optional input of type int32, specifying the number of valid ROIs. This parameter is reserved . \n

* @par Attributes:
* @li spatial_scale: A required attribute of type float32, specifying the scaling ratio of "features" to the original image.
* @li pooled_height: A required attribute of type int32, specifying the H dimension.
* @li pooled_width: A required attribute of type int32, specifying the W dimension.
* @li sample_num: An optional attribute of type int32, specifying the horizontal and vertical sampling frequency of each output. If this attribute is set to "0",
* the sampling frequency is equal to the rounded up value of "rois", which is a floating point number. Defaults to "2".
* @li roi_end_mode: An optional attribute of type int32, specifying the align mode. Defaults to "1", supports 0/1/2/3. \n
* "0" is compatible with align = False for all frameworks. \n 
* "1" is compatible with align = True for TensorFlow. \n
* "2" is compatible with align = True for pyTorch. \n 
* "3" is compatible with align = True for MmDetecion v0.6. \n
* @li pool_mode: An optional attribute of type string, specifying the pooling mode. Defaults to "avg", supports "avg" and "max". \n

* @par Outputs:
* y: Outputs the feature sample of each ROI position. The format is 5HD Tensor of type float32 or float16.
* The axis N is the number of input ROIs. Axes H, W, and C are consistent
* with the values of "pooled_height",
* "pooled_width", and "features", respectively.
*/
REG_OP(ROIAlign)
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(rois_n, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(spatial_scale, Float)
    .REQUIRED_ATTR(pooled_height, Int)
    .REQUIRED_ATTR(pooled_width, Int)
    .ATTR(sample_num, Int, 2)
    .ATTR(roi_end_mode, Int, 1)
    .ATTR(pool_mode, String, "avg")
    .OP_END_FACTORY_REG(ROIAlign)

/**
*@brief Greedily selects a subset of bounding boxes in descending order of
score . \n

*@par Inputs:
*Input boxes and  scores must be float16 type. Inputs include:
*@li boxes: A input tensor with shape [num_batches,spatial_dimension,4].
The single box data format is indicated by center_point_box.
*@li scores: A input tensor with shape [num_batches,num_classes,spatial_dimension]
*@li max_output_size: A scalar integer tensor representing the maximum number
of boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding
whether boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for
deciding when to remove boxes based on score . \n

*@par Attributes:
*center_point_box:Integer indicate the format of the box data.
The default is 0. 0 - the box data is supplied as [y1, x1, y2, x2]
where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair
of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.Mostly used for TF models.
1 - the box data is supplied as [x_center, y_center, width, height].
 Mostly used for Pytorch models. \n

*@par Outputs:
*@li selected_indices: A 2-D integer tensor of shape [M] representing the
selected indices from the boxes tensor, where M <= max_output_size. \n

*@attention Constraints:
*Input boxes and  scores must be float16 type . \n

*@par Third-party framework compatibility
*Compatible with onnx NonMaxSuppression operator.

*@par Restrictions:
*Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(NonMaxSuppressionV6)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(max_output_size, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(iou_threshold, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(score_threshold, TensorType({DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .ATTR(center_point_box, Int, 0)
    .ATTR(max_boxes_size, Int, 0)
    .OP_END_FACTORY_REG(NonMaxSuppressionV6)

/**
*@brief Performs SSD prior box detection . \n

*@par Inputs:
* Two inputs, including:
*@li x: An NCHW feature map of type is float32 or float16.
*@li img: source image. Has the same type and format as "x" . \n

*@par Attributes:
*@li min_size: A required float32, specifying the minimum edge length of a square prior box.
*@li max_size: A required float32, specifying the maximum edge length of a square prior box: sqrt(min_size * max_size)
*@li aspect_ratio: An required float32, specifying the aspect ratio for generated rectangle boxes. The height
is min_size/sqrt(aspect_ratio), the width is min_size*sqrt(aspect_ratio). Defaults to "1.0".
*@li img_h: An optional int32, specifying the source image height. Defaults to "0".
*@li img_w: An optional int32, specifying the source image width. Defaults to "0".
*@li step_h: An optional float32, specifying the height step for mapping the center point from the feature map to the source image. Defaults to "0.0".
*@li step_w: An optional float32, specifying the width step for mapping the center point from the feature map to the source image. Defaults to "0.0".
*@li flip: An optional bool. If "True", "aspect_ratio" will be flipped. Defaults to "True".
*@li clip: An optional bool. If "True", a prior box is clipped to within [0, 1]. Defaults to "False".
*@li offset: An optional float32, specifying the offset. Defaults to "0.5".
*@li variance: An optional float32, specifying the variance of a prior box, either one or four variances. Defaults to "0.1" (one value) . \n

*@par Outputs:
*y: An ND tensor of type float32 or float16, specifying the prior box information, including its coordinates and variance . \n

*@attention Constraints:
* This operator applies only to SSD networks.
*@see SSDDetectionOutput()
*@par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(PriorBox)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(img, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(min_size, ListFloat)
    .REQUIRED_ATTR(max_size, ListFloat)
    .REQUIRED_ATTR(aspect_ratio, ListFloat)
    .ATTR(img_h, Int, 0)
    .ATTR(img_w, Int, 0)
    .ATTR(step_h, Float, 0.0)
    .ATTR(step_w, Float, 0.0)
    .ATTR(flip, Bool, true)
    .ATTR(clip, Bool, false)
    .ATTR(offset, Float, 0.5)
    .ATTR(variance, ListFloat, {0.1})
    .OP_END_FACTORY_REG(PriorBox);
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_

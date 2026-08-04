/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2023. All rights reserved.
 * Description: detection_defs
 */
#ifndef INC_API_GRAPH_OP_DETECTION_DEFS_H
#define INC_API_GRAPH_OP_DETECTION_DEFS_H
#include "graph/operator_hiai_reg.h"

// clang-format off
namespace hiai {
/*
 * Permutes the dimensions of the input according to a given pattern.
 * <Input>
 *    x : Input tensor.
 * <Output>
 *    y : Has the same shape as the input, but with the dimensions re-ordered according to the specified pattern.
 * <Attr>
 *    order : Tuple of dimension indices indicating the permutation pattern, list of dimension indices.
 *            When order is -1, it means reverse order.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto permute = hiai::op::Permute("permute")
 *                   .set_input_x(x)
 *                   .set_attr_order({0, 3, 1, 2});
 */
HIAI_REG_OP(Permute)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL }))
.HIAI_ATTR(order, AttrValue::LIST_INT ({ 0 }))
.HIAI_OP_END()

/*
 * A layer in SSD net, the role of SSDDetectionOutput is to generate the  number and coordinate of label boxes
 * according to position offset of prior box and adjusting parameters(nms threshold and confidence threshold).
 * only support in CPUCL
 * <Input>
 *    bbox_delta : Frame position offset data.
 *    score : Confidence data.
 *    anchors : Preselection box data.
 * <Output>
 *    out_boxnum : The number of output box.
 *    y : Output box data.
 * <Attr>
 *    num_classes : The number of classes.
 *    share_location : if true, bounding box are shared among different classes.
 *    background_label_id : Background label id. If there is no background class,set it as -1.
 *    iou_threshold : Non maximum suppression threshold and must be between 0 and 1.
 *    top_k : Number of bboxes to be considered for per class before NMS.
 *    eta : Parameter for adaptive nms.
 *    variance_encoded_in_target : If true, variance is encoded in target;
 *    code_type : Type of coding method for bbox.
 *    keep_top_k : Number of total bboxes to be kept per image after nms step.
 *    confidence_threshold : Confidence threshold parameter, and must be between 0 and 1.
 * <Added in HiAI version>
 *    100.500.010.010
 */
HIAI_REG_OP(SSDDetectionOutput)
.HIAI_INPUT(bbox_delta, TensorType({ DT_FLOAT }))
.HIAI_INPUT(score, TensorType({ DT_FLOAT }))
.HIAI_INPUT(anchors, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(out_boxnum, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(num_classes, AttrValue::INT)
.HIAI_ATTR(share_location, AttrValue::BOOL { true })
.HIAI_ATTR(background_label_id, AttrValue::INT { 0 })
.HIAI_ATTR(iou_threshold, AttrValue::FLOAT { 0.3f })
.HIAI_ATTR(top_k, AttrValue::INT { 200 })
.HIAI_ATTR(eta, AttrValue::FLOAT { 1.0f })
.HIAI_ATTR(variance_encoded_in_target, AttrValue::BOOL { false })
.HIAI_ATTR(code_type, AttrValue::INT { 1 })
.HIAI_ATTR(keep_top_k, AttrValue::INT { 200 })
.HIAI_ATTR(confidence_threshold, AttrValue::FLOAT { 0.0f })
.HIAI_OP_END()
} // namespace hiai
// clang-format on

#endif
/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
 * \file selection_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Extracts a slice from a tensor.
*       This operation extracts a slice of size "size" from a tensor "x"
*       starting at the location specified by "offsets".

* @par Inputs:
* @li x: A Tensor. Must be one of the following types:
* bfloat16, float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, hifloat8, float8_e5m2, float8_e4m3fn.
* @li offsets: A Tensor of type int32 or int64. The starting location for the slice.
* @li size: A Tensor of type int32 or int64. The tensor size for the slice. \n

* @attention Constraints:
* @li 0 <= offset[i] <= offset[i] + size[i] <= x_dim[i] for i in [0,n],
* n is the dimension of the tensor "x". \n
* @li offsets, size and x must have the same rank.

* @par Outputs:
* y: A Tensor. Has the same type as "x". The slice extracted from the tensor. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Slice.
*/
REG_OP(Slice)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(offsets, TensorType::IndexNumberType())
    .INPUT(size, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Slice)

/**
* @brief Extracts a strided slice of a tensor. Roughly speaking, this op
*   extracts a slice of size (end-begin)/stride from the given input tensor.
*   Starting at the location specified by begin the slice continues by
*   adding stride to the index until all dimensions are not less than end. \n
*
* @par Inputs:
* Five inputs, including:
* @li x: A Tensor. Must be one of the following types:
* double, float32, float16, bfloat16, complex32, complex64, complex128,
* int8, uint8, int16, uint16, int32, uint32, int64, uint64, qint8, quint8, qint16, quint16, qint32, bool.
* @li begin: A Tensor of type int32 or int64, for the index of the first value to select.
* @li end: A Tensor of type int32 or int64, for the index of the last value to select.
* @li axes: A Tensor of type int32 or int64, indicate axis to be select.
* @li strides: A Tensor of type int32 or int64, for the increment. \n
*
* @par Attributes:
* @li begin_mask: A Tensor of type int32.
*     Developers can ignore this attribute.
*     A bitmask where a bit "i" being "1" means to ignore the begin
*     value and instead use the largest interval possible.
* @li end_mask: A Tensor of type int32.
*     Developers can ignore this attribute.
*     Analogous to "begin_mask".
* @li ellipsis_mask: A Tensor of type int32.
*     Developers can ignore this attribute.
*     A bitmask where bit "i" being "1" means the "i"th position
*     is actually an ellipsis.
* @li new_axis_mask: A Tensor of type int32.
*     Developers can ignore this attribute.
*     A bitmask where bit "i" being "1" means the "i"th
*     specification creates a new shape 1 dimension.
* @li shrink_axis_mask: A Tensor of type int32.
*     Developers can ignore this attribute.
*     A bitmask where bit "i" implies that the "i"th
*     specification should shrink the dimensionality. \n
*
* @par Outputs:
* y: A Tensor that has the same type as "x", but except bool.
*
* @attention Constraints:
*
* @par Third-party framework compatibility
* Compatible with the onnx operator Slice.
*/
REG_OP(StridedSliceV2)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL}))
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(strides, TensorType::IndexNumberType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(StridedSliceV2)

/**
* @brief Creates a new tensor by applying sparse "x" to individual values or slices within a tensor
* (initially zero for numeric, empty for string) of the given "shape" according to "indices".

* @par Inputs:
* @li indices: The index tensor. Format is ND. Support 1D ~ 8D. Must be one of the following types: int32, int64.
* @li x: The source tensor. Format is ND. Type must be the BasicType. Support 1D ~ 8D.
* @li shape: The shape of "y". Format is ND. Support 1D ~ 8D. Must be one of the following types: int32, int64.

* @par Outputs:
* y: A output tensor with same type as input "x".

* @attention Constraints:
* @li indices.shape[-1] <= shape.rank, where the range of shape.rank is [1, 7]
* @li x.shape = indices.shape[:-1] + shape[indices.shape[-1]:].

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNd.
*/
REG_OP(ScatterNd)
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(x, TensorType::BasicType())
    .INPUT(shape, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(ScatterNd)

/**
* @brief Computes the cumulative sum of the tensor "x" along "axis" .

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types:
* int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32,
* double, complex64, complex128, bfloat16.
* @li axis: A Tensor of type int32 or int64. Range is [-rank(x),rank(x)). Dim and shape must be 1.
*
* @par Attributes:
* @li exclusive: A bool. Defaults to "False". If "False", performs inclusive cumsum, which means that the first element
* of the input is identical to the first element of the output. If "True", performs exclusive cumsum.
* @li reverse: A bool. Defaults to "False". If "True", the cumulative sum is calculated from the end of the 
* tensor towards the beginning. If "False", the cumulative sum is calculated from the beginning of the tensor towards
* the end.
*
* @par Outputs:
* y: A Tensor. Has the same type and shape as "x".
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Cumsum.
*/
REG_OP(Cumsum)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(Cumsum)

/**
* @brief Crops the input tensor x to the shape of size. For example:
* (1) x: bottom to be cropped, with shape (20, 50, 512, 512);
* (2) size: reference input for cropping, with shape (20, 10, 256, 256);
* (3) axis = 1;
* (4) offsets = (25, 128, 128);
* (5) y = x[:, 25:25 + size.shape[1], 128:128 + size.shape[2], 128:128 +
* size.shape[3]] .

* @par Inputs:
* Inputs include:
* @li x: A required Tensor. Must be one of the following types: float16,
* float32, int8, uint8, int16, uint16, int32, uint32,int64, uint64.
* The format support ND, NCHW, NHWC and NC1HWC0. Shape support 1D ~ 8D.
* @li size: A required Tensor. Must be one of the following types: float16,
* float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64.
* The format support ND, NCHW, NHWC and NC1HWC0. Shape support 1D ~ 8D.
* The format and type are same as "x".
* Each dimension of "size" cannot exceed the corresponding dimension of "x". \n

* @par Attributes:
* @li axis: A required int, specifying the first dimension to crop. Defaults
* to "2". When ori_format of x is equal to "NCHW", the ori_shape of x is
* equal to 4 and the axis is greater than or equal to 2.
* The Op Crop can support HC1HWC0 and ND.
* @li offsets: A required array,
* specifying the shift for all/each dimension to align the cropped bottom with
* the reference bottom. No default value.
* Must be one of the following types: float16, float32, int8, uint8, int16,
* uint16, int32, uint32, int64, uint64. \n

* @par Outputs:
* y: A required Tensor. The format support ND, NCHW, NHWC and NC1HWC0.
* Shape support 1D ~ 8D. Must be one of the following types: float16,
* float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64.
* Has the same type, format and shape as "size" . \n

* @attention Constraints:
* @li "y" must have the same type and shape as "size". "x" must have the same
* type as "size".
* @li "axis" must be less than the rank of "x".
* @li The "offsets" for each dimension must not exceed the maximum value of
* the corresponding dimension of "x".
* @li The array length of "offsets" plus the value of "axis" equals to the
* rank of "y".
* @li When ori_format of x is equal to "NCHW", the ori_shape of x is
* equal to 4 and the axis is greater than or equal to 2.
* The Op Crop can support HC1HWC0 and ND.
* @par Third-party framework compatibility
* Compatible with the Caffe operator Crop.
*/
REG_OP(Crop)
      .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
      .INPUT(size, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
      .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
      .ATTR(axis, Int, 2)
      .REQUIRED_ATTR(offsets, ListInt)
      .OP_END_FACTORY_REG(Crop)

/**
* @brief Gather slices from "params" according to "indices"."indices" must be
    an integer tensor of any dimension(usually 0-D or 1-D).
    Produces an output tensor with shape "indices.shape + params.shape[1:]" .

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: complex128, complex64, float64, float32, float16,
*     int16, int32, int64, int8, qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
*     bool, bfloat16.
* @li indices: A Tensor of type int32 or int64 .

* @par Attributes:
* @li validate_indices: Whether to verify the values of indices, not enabled currently.
* @li batch_dims: An optional int. Defaults to 0.
* @li is_preprocessed: An optional bool. Whether to preprocess. Defaults to false.
* @li negative_index_support: An optional bool. Defaults to false.

* @par Outputs:
* y: A Tensor. Has the same type as "x" .

* @attention Constraints:
* "indices" is in the range [0, x.shape[0]) .

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Gather .

*/
REG_OP(Gather)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_BF16}))
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_BF16}))
    .ATTR(validate_indices, Bool, true)
    .ATTR(batch_dims, Int, 0)
    .ATTR(is_preprocessed, Bool, false)
    .ATTR(negative_index_support, Bool, false)
    .OP_END_FACTORY_REG(Gather)

/**
* @brief Gather slices from "x" according to "indices" by corresponding axis, produces a output tensor
* with shape(x.shape[:axis]+indices.shape[batch:]+x.shape[axis+1:]). When the impl_mode is set
* as "support out of bound index", if the indices data is out of bound, the corresponding results
* will be set as 0. Otherwise, an aic_error will occur.

* @par Inputs:
* @li x: A ND(Support 1D~8D) Tensor. Must be one of the following types: complex128, complex64, float64, float32, float16,
*     int16, int32, int64, int8, qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
*     bool, string, bfloat16.
* @li indices: A ND(Support 1D) Tensor of type int32 or int64.
* @li axis: A Scalar with type as int32 or int64. Must be in the range [-rank(input_tensor), rank(input_tensor)).

* @par Attributes:
* @li batch_dims: An optional int which means the number of data to be deal with. Defaults to 0.
* @li is_preprocessed: An optional bool. Whether to preprocess, wihch is true means need to be preprocess and false means not. Defaults to false.
* @li negative_index_support: An optional bool, which is true means support index is negative, and false means not. Defaults to false.

* @par Outputs:
* y: A ND Tensor which has the same type as "x".

* @attention Constraints:
* @li Value in indices must be in range [0, x.shape[axis]).
* @li Default mode is HIGH_PERCISION.
      Only HIGH_PERCISION mode support negative index, and negative index in HIGH_PERFORMANCE mode may cause precision abnormal or aicore error.
* @li Batch_dims must be in the range [max(-rank(input_tensor),-rank(indices)), min(rank(input_tensor), rank(indices))).
* @li (batch_dims + rank(input_tensor)) % rank(input_tensor) must be less than or equal to (axis + rank(input_tensor)) % rank(input_tensor).
* @li The first batch_dims dimensions of params and indices are same.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator GatherV2 .

*/
REG_OP(GatherV2)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_STRING, DT_BF16}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_STRING, DT_BF16}))
    .ATTR(batch_dims, Int, 0)
    .ATTR(is_preprocessed, Bool, false)
    .ATTR(negative_index_support, Bool, false)
    .OP_END_FACTORY_REG(GatherV2)

/**
* @brief Gather slices from "x" into a tensor with shape specified by
* "indices". "indices" is an K-dimensional integer tensor, best thought of as a
* (K-1)-dimensional tensor of "indices" into "params", where each element
* defines a slice of "params":
*   output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]
* "indices" defines slices into the first N dimensions of
* "params", where
*           N = indices.shape[-1]
*     indices = [[0, 0], [1, 1]]
*      x = [['a', 'b'], ['c', 'd']]
*      output = ['a', 'd']
* When the impl_mode is set as "support out of bound index", if the indices
* data is out of bound, the corresponding results will be set as 0. Otherwise,
* an aic_error will occur.

* @par Inputs:
* @li x: A ND(Support 1D~8D) Tensor. Must be one of the following types:
*     complex128, complex64, float64, float32, float16, int16, int32, int64,
*     int8, qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64,
*     uint8, bool, string, bfloat16.
* @li indices: A ND(Support 1D) Tensor of type int32 or int64.

* @par Attributes:
* negative_index_support: An optional bool. Defaults to false.

* @par Outputs:
* y: A ND(Support 1D~8D) Tensor which has the same type as "x".


* @par Third-party framework compatibility
* Compatible with the TensorFlow operator GatherNd.
*/
REG_OP(GatherNd)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_STRING, DT_BF16}))
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                          DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32,
                          DT_UINT64, DT_UINT8, DT_BOOL, DT_STRING, DT_BF16}))
    .ATTR(negative_index_support, Bool, false)
    .OP_END_FACTORY_REG(GatherNd)

/**
* @brief Constructs a tensor by tiling a given tensor .

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor.
* Must be one of the following types: DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
 DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
 DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_BOOL,DT_HIFLOAT8, DT_FLOAT8_E5M2, 
 DT_FLOAT8_E4M3FN
* @li multiples: A 1D Tensor of type int32 or int64.
*     The length must be the same as the number of dimensions in "input"

* @par Outputs:
* y: A Tensor. Has the same type as "x" . \n

* @see TileD()

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Tile.
*/
REG_OP(Tile)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                          DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                          DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(multiples, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                           DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                           DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Tile)

/**
* @brief Select elements from "then" or "else", depending on "condition" .

* @par Inputs:
* Three inputs, including:
* @li condition: A tensor of type bool. If condittion is true, outputs will be set as then. If condittion is false, outputs will be set as else.
* @li then: A tensor. Must be one of the following types: float16, float32, double, int8, int16, int32, int64, 
 *uint8, uint16, uint32, uint64, complex64, complex128, bool, bfloat16
* @li else: A tensor of the same type as "then" . \n

* @par Outputs:
* result: A tensor. Has the same type as "then" . \n

* @attention Constraints:
* @li The input tensors of condition, then and else must meet the broadcast relationship.
* @li The shape of result is formed by broadcasting condition, then and else.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SelectV2.
*/
REG_OP(SelectV2)
    .INPUT(condition, TensorType({DT_BOOL}))
    .INPUT(then,TensorType({DT_COMPLEX128,DT_COMPLEX64,DT_DOUBLE,DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT16,DT_UINT32,DT_UINT64,DT_UINT8,DT_BOOL,DT_BF16}))
    .INPUT(else,TensorType({DT_COMPLEX128,DT_COMPLEX64,DT_DOUBLE,DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT16,DT_UINT32,DT_UINT64,DT_UINT8,DT_BOOL,DT_BF16}))
    .OUTPUT(result,TensorType({DT_COMPLEX128,DT_COMPLEX64,DT_DOUBLE,DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT16,DT_UINT32,DT_UINT64,DT_UINT8,DT_BOOL,DT_BF16}))
    .OP_END_FACTORY_REG(SelectV2)

/**
* @brief Returns a one-hot tensor. The locations represented by index in "x" take value "on_value",
*         while all other locations take value "off_value" .

* @par Inputs:
* Four inputs, including:
* @li x: A 1-7D tensor of indices, format supports ND, and data type must be one of the following types: int32, uint8, int64.
* @li depth: A scalar which is the depth of the one hot dimension, format supports ND, and data type must be int32 or int64
*     Its shape can be 1-8D, but only the first element make sense.
* @li on_value: A scalar. The value to fill in output when indices[j] = i, format supports ND.
*     Must be one of the following types: float16, float32, int64, int32, int8, uint8.
*     Its shape can be 1-8D, but only the first element make sense.
* @li off_value: A scalar. The value to fill in output when indices[j] != i, format supports ND.
*     Has the same type as "on_value". Its shape can be 1-8D, but only the first element make sense.

* @par Attributes:
* axis: The axis to fill. An int with a minimum value of -1 and a maximum value of dims of x. Defaults to "-1"

* @par Outputs:
* y: A 1-8D tensor. Has the same type as "on_value" . \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator OneHot.
*/
REG_OP(OneHot)
    .INPUT(x, TensorType({DT_UINT8, DT_INT32, DT_INT64}))
    .INPUT(depth, TensorType({DT_INT32, DT_INT64}))
    .INPUT(on_value, TensorType::BasicType())
    .INPUT(off_value, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(OneHot)

/**
* @brief Finds values and indices of the "k" largest elements for the last
* dimension . \n

* @par Inputs:
* Two inputs, including:
* @li x: A 1D or higher tensor of type RealNumberType, with the last dimension
* at least "k".
* @li k: A 0D Tensor of type int32.
* Number of top elements to look for along the last dimension (along each row
* for matrices) . \n

* @par Attributes:
* @li sorted: An optional bool. Defaults to "True".
* If "True", the returned "k" elements are themselves sorted.
* If "False", the returned "k" elements are not sorted.
* @li largest: An optional bool, controls whether to return largest or smallest elements. Defaults to true.
* If "True", the "k" largest elements are returned in descending order.
* If "False", the "k" smallest elements are returned in ascending order.
* @li dim: An optional int. Default is -1. 0-D. Number of top elements to look for along the last dimension (along each row for matrices). \n

* @par Outputs:
* @li values: A Tensor, specifying the sorted data. Has the same type as
* "x".
* @li indices: A Tensor of type int32, specifying the indices of sorted data . \n

* @see TopK()
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator TopKV2.
*/
REG_OP(TopK)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .ATTR(sorted, Bool, true)
    .ATTR(largest, Bool, true)
    .ATTR(dim, Int, -1)
    .OP_END_FACTORY_REG(TopK)

/**
* @brief Finds values and indices of the "k" largest elements for the last
* dimension . \n

* @par Inputs:
* Two inputs, including:
* @li x: A 1D-8D tensor, with the last dimension at least "k".
* Supported type: float16, float32, int16, int8, uint8, int32, int64, bfloat16, uint32, uint16, uint64. 
* Supported format: ND.
* @li k: A 0D Tensor of type int32. Supported format: ND.
* Number of top elements to look for along the last dimension (along each row
* for matrices) . \n

* @par Attributes:
* @li sorted: An optional bool. Defaults to "True".
* If "True", the returned "k" elements are themselves sorted.
* If "False", the returned "k" elements are not sorted.
* @li dim: An optional int. Defaults to -1. For reserved use.
* @li largest: An optional bool, controls whether to return largest or smallest elements. Defaults to true.
* If "True", the "k" largest elements are returned in descending order.
* If "False", the "k" smallest elements are returned in ascending order. \n

* @par Outputs:
* @li values: A Tensor, specifying the sorted data. Has the same type and format as
* "input".
* @li indices: A Tensor of type int32, specifying the indices of sorted data. Supported format: ND . \n

* @see TopK()
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator TopKV2.
*/
REG_OP(TopKV2)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .ATTR(sorted, Bool, true)
    .ATTR(dim, Int, -1)
    .ATTR(largest, Bool, true)
    .OP_END_FACTORY_REG(TopKV2)

/**
* @brief Performs object detection . \n

* @par Inputs:
* @li cls_prob: An NCHW tensor of type float16 or float32,
* specifying the probability of the proposal is the background class.
* @li bbox_delta: An NCHW tensor of type float16 or float32, specifying the coordinates of the proposals bounding boxes.
* @li im_info: An ND tensor of type float16 or float32, specifying the Image information . \n

* @par Attributes:
* @li feat_stride: A optional float32, specifying the stride of the sliding window.
* Must be greater than "0".Defaults to "16".
* @li base_size: A optional float32, specifying the size of the generated base box.
* Must be greater than "0". Defaults to "16".
* @li min_size: A optional float32, specifying the minimum edge length of a proposal.
* A box with any edge less than this value is removed. Must be greater than "0". Defaults to "16".
* @li ratio: A optional list of floats, specifying the aspect ratio of the generated base box. Defaults to [0.5, 1, 2].
* @li scale: A optional list of floats, specifying the ratio of the size of the generated base box to "base_size".
* Defaults to [8, 16, 32].
* @li pre_nms_topn: A required int, specifying top K boxes before NMS.
* For float16 input, pre_nms_topn <= 6000. For float32 input, pre_nms_topn <= 3000. Defaults to "3000".
* @li post_nms_topn: A required int, specifying the number of boxes to be output after NMS.
* The value is a multiple of 16. For float16 input, post_nms_topn <= 6000. For float32 input,
* post_nms_topn <= 3000 (the maximum multiple of 16 is 2992 within the range). Defaults to "304".
* @li iou_threshold: A required float32, specifying the NMS threshold. The value range is (0,1]. Defaults to "0.7".
* @li output_actual_rois_num: An optional bool. Defaults to "false" . \n

* @par Outputs:
* @li rois: A Tensor with shape [batch, 5, post_nms_topn],
* of type float16 or float32, specifying the output box information.
* "post_nms_topn" must be a multiple of 16. The dimension "5" indicates (batchID, x1, y1, x2, y2).
* The number of BBoxes output per batch is determined by "actual_rois_num".
* @li actual_rois_num: A Tensor with shape [batch, 8], of type int32, specifying the number of BBoxes output per batch.
* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
 REG_OP(Proposal)
     .INPUT(cls_prob, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(bbox_delta, TensorType({DT_FLOAT16, DT_FLOAT}))
     .INPUT(im_info, TensorType({DT_FLOAT16, DT_FLOAT}))
     .OUTPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
     .OUTPUT(actual_rois_num, TensorType({DT_INT32}))
     .ATTR(feat_stride, Float, 16)
     .ATTR(base_size, Float, 16)
     .ATTR(min_size, Float, 16)
     .ATTR(ratio, ListFloat, {0.5, 1, 2})
     .ATTR(scale, ListFloat, {8, 16, 32})
     .ATTR(pre_nms_topn, Int, 3000)
     .ATTR(post_nms_topn, Int, 304)
     .ATTR(iou_threshold, Float, 0.7)
     .ATTR(output_actual_rois_num, Bool, false)
     .OP_END_FACTORY_REG(Proposal)
/**
* @brief Creates a sequence of numbers . \n

* @par Inputs:
* Three inputs, including:
* @li start: A 0D tensor (scalar). Acts as first entry in the range if "limit"
*   is not "None"; otherwise, acts as range limit and first entry defaults to "0".
*   The supported types are:float16, float32, int32, double, int64, bfloat16, format supports ND.
* @li limit: A 0D tensor (scalar). Upper limit of sequence, exclusive. If "None",
*   defaults to the value of "start" while the first entry of the range
*   defaults to "0". The supported types are:float16, float32, int32, double, int64, bfloat16, format supports ND.
* @li delta: A 0D tensor (scalar). Number that increments "start".
*   Defaults to "1". The supported types are:float16, float32, int32, double, int64, bfloat16, format supports ND. \n
*
* @par Outputs:
* y: A 1D tensor which is the sequence of numbers, format supports ND.
*    The supported types are:float16, float32, int32, double, int64, bfloat16. \n
*    The types of start/limit/delta and output should be the same when the dtypes are:float16, bfloat16, int64. \n
*    The auto inferred type of output is the same as input when the types of start/limit/delta are the same,
*    otherwise the auto inferred type of output is float32. \n
*    The double type of y is not supported in Ascend910_95 AI Processor. \n
*
* @par Attributes:
* is_closed: An optional attribute of type bool, inducating upper limit is closed or not. 
* If true, upper limit is closed. If false, upper limit is opened. The default value is false.
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Range or PyTorch operator Range/Arange.
*/
REG_OP(Range)
    .INPUT(start, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .INPUT(limit, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .INPUT(delta, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .ATTR(is_closed, Bool, false)
    .OP_END_FACTORY_REG(Range)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_

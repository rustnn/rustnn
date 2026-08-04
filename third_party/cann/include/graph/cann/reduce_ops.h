/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file reduce_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Compute reduction on dimensions specified by "axis".
*Four reduction operations are provided:
*SUM     Computes the sum of elements across specified dimensions of a tensor.
*ASUM    Computes the sum of absolute values of elements across specified
*dimensions of a tensor.
*SUMSQ   Computes the sum of squares of elements across specified
*dimensions of a tensor.
*SUMSQ   Computes the mean values of elements across specified
*dimensions of a tensor .

*@par Inputs:
*x: A Tensor of type float16 or float32. Support 1D ~ 8D, Support format:["ND", "NC1HWC0"].

*@par Attributes:
*@li operation: An optional int32 from 1(SUM), 2(ASUM), 3(SUMSQ), and 4(MEAN),
*specifying the reduction algorithm. Defaults to "1".
*@li axis: An optional int32, specifying the first axis to reduce.
*Defaults to "0".
*The value range is [-N, N-1], where N is the input tensor rank.
*@li coeff: An optional float32, specifying the scale coefficient.
*Defaults to "1.0" . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x". Support 1D ~ 3D, Support format:["ND", "NC1HWC0"].

*@attention Constraints: The Reduction operator supports type float16
*only on the device chip.
*@par Third-party framework compatibility
* Compatible with the Caffe operator Reduction.
*/
REG_OP(Reduction)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(operation, Int, 1)
    .ATTR(axis, Int, 0)
    .ATTR(coeff, Float, 1.0)
    .OP_END_FACTORY_REG(Reduction);

/**
* @brief Reduces "x" along the dimensions according to "axis".

* @par Inputs:
* Two inputs, including:
* @li x: A tensor. Must be one of the following types:
*complex128, complex64, double, float32, float16, int64, int32, int16, int8,
*uint64, uint32, uint16, uint8, bfloat16. The data format supports ND.
* @li axes: The dimensions to reduce. Must be one of the following types:
* int, list, tuple, NoneType. Data type must be int32 or int64.
* If None (the default), reduces all dimensions.
* Must be in the range [-rank(x), rank(x)).

* @par Attributes:
* @li keep_dims: An optional bool. Defaults to false.
* If true, retains reduced dimensions with length 1.
* If false, the rank of the tensor is reduced by 1 for each entry in axis.
* @li noop_with_empty_axes: An optional bool. Defaults to true.
* If true, when axes = [], not reduce.
* If false, when axes = [], reduce all.
* @par Outputs:
* y: A tensor. Has the same type and format as "x".

* @attention Constraints:
* @li When converting ONNX to OM, if the axes of the ReduceMean operator is empty,
   and noop_with_empty_axes is true, it is recommended to use the mean function with dim explicitly
   set to all axes(e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator ReduceMean.
*/
REG_OP(ReduceMean)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMean)

/**
*@brief  Reduce a tensor on a certain axis based on product.

*@par Inputs:
*Two inputs, including:
*@li x: A Tensor. Must be the type of NumberType.(NumberType
*includes: complex128, complex64, double, float32, float16, int16,
*int32, int64,int8, qint32, qint8, quint8, uint16, uint32, uint64,
*uint8, bfloat16, complex32).Supported format list ["ND"].
*@li axes: A Tensor. Must be the type of IndexNumberType(
*includes: int32, int64). The dimensions to reduce.Supported format list ["ND"]. \n

*@par Attributes:
*keep_dims: A bool. If true, retains reduced dimensions with length 1.
*Optional and defaults to "False" . \n
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend910_95 AI Processors and later products.

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceProd.
*/
REG_OP(ReduceProd)
    .INPUT(x,TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y,TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceProd)

/**
* @brief Computes the sum of elements across dimensions of a tensor.

* @par Inputs:
* Two inputs, including:
* @li x: A tensor. Must be one of the following types:
* complex128, complex64, double, float32, float16, int16, int32, int64,
* int8, qint32, qint8, quint8, uint16, uint32, uint64, uint8, bfloat16,
* complex32.
* @li axes: A 1D list or tuple of IndexNumberType(int32 or int64).
* Specifies the dimensions to reduce.

* @par Attributes:
* keep_dims: An optional bool. If "true", retains reduced dimensions with
* length 1. Defaults to "false".
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend910_95 AI Processors and later products.

* @par Outputs:
* y: The reduced tensor. Has the same type and format as input "x".

* @attention Constraints:
* @li The value range of "axes" is [-dims, dims - 1]. "dims"
  indicates the dimension length of "x".
* @li When converting ONNX to OM, if the axes of the ReduceSum operator is empty,
   it is recommended to use the sum function with dim explicitly set to all axes
   (e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Sum.
*/
REG_OP(ReduceSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceSum)

/**
* @brief Returns the maximum of elements across dimensions of a Tensor .

* @par Inputs:
* Two inputs, including:
* @li x: A multi-dimensional Tensor of Must be the type of NumberType. Supported format list ["ND"]
* @li axes: A Scalar of type in IndexNumberType(IndexNumberType includes the
  following types: int32, int64.), specifying the axes information
  of the index with the maximum value. Supported format list ["ND"] \n

* @par Attributes:
* keep_dims: A bool, specifying whether to keep dimensions for the output Tensor.
* Optional and defaults to "false". \n
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend910_95 AI Processors and later products.

* @par Outputs:
* y: A multi-dimensional Tensor, specifying the maximum value of the
  corresponding axis in the tensor.
  Has the same type as "x". (If "keep_dims" is set to "false",
  the output dimensions are reduced by "dimension" compared with that of "x".
  Otherwise, the output has one fewer dimension than "x").Supported format list ["ND"]

* @attention Constraints:
* @li The value range of "axes" is [-dims, dims - 1]. "dims"
  indicates the dimension length of "x".
* @li When converting ONNX to OM, if the axes of the ReduceMax operator is empty,
   it is recommended to use the amax function with dim explicitly set to all axes
   (e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Max.
*/
REG_OP(ReduceMax)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMax)

/**
* @brief Computes the minimum of elements across dimensions of a tensor .

* @par Inputs:
* @li x: A tensor. Must be the type of NumberType.
* @li axes: A tensor of type of IndexNumberType.(IndexNumberType
* includes: int32, int64.) Specifies the dimensions to reduce.
* Defaults to "None".

* @par Attributes:
* keep_dims: An optional bool. If "True", reduced dimensions will be retained.
* Defaults to "False".
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend910_95 AI Processors and later products.

* @par Outputs:
* y: A tensor. Must be the type of NumberType.

* @attention Constraints:
* @li If "axes = None", all dimensions will be reduced. "axes" must be in the
  range [-rank(input_shape), rank(input_shape)).
* @li When converting ONNX to OM, if the axes of the ReduceMin operator is empty,
   it is recommended to use the amin function with dim explicitly set to all axes
   (e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator reduce_min.
*/
REG_OP(ReduceMin)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMin)

/**
* @brief Computes the log and sum and exp of elements across dimensions of a tensor.
* Reduces "x" along the dimensions given in "axes".
* Unless "keep_dims" is true, the rank of the tensor is reduced by 1 for each
* entry in "axes". If "keep_dims" is true, the reduced dimensions
* are retained with length 1.
*
* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float32, float16, bfloat16.
* @li axes: A 1D list or tuple of int32 or int64. Specifies the dimensions to reduce. \n
*
* @par Attributes:
* keep_dims: An optional bool. If "true", retains reduced dimensions with length 1. Defaults to "false" . \n
*
* @par Outputs:
* y: The reduced tensor. Has the same type and format as input "x" . \n
*
* @par Third-party framework compatibility
* Compatible with the Onnx operator ReduceLogSumExp.
*/
REG_OP(ReduceLogSumExp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(axes, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceLogSumExp)
} //namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_REDUCE_OPS_H_

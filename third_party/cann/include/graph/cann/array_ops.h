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
 * \file array_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ARRAY_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ARRAY_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Reshapes a tensor. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*@li x: A tensor. All data types are supported. \n
*@li shape: A tensor. Must be one of the following types: int32, int64. Defines the shape of the output tensor. \n

*@par Attributes:
*@li axis: An optional int32 or int64. The first dimension to reshape. Defaults to "0".
*@li num_axes: An optional int32 or int64. The extent of the reshape. Defaults to "-1". \n

*@par Outputs:
*y: A tensor. The same type as input x. \n

*@attention Constraints:
*This operator cannot be directly called by the acllopExecute API. \n

*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator Reshape.
*@li Compatible with the Caffe operator Reshape.
*/
REG_OP(Reshape)
    .INPUT(x, TensorType::ALL())
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Reshape)

/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*x: Original tensor. All data types are supported. \n

*@par Attributes:
*axes: List of ints indicating the dimensions to be inserted. Defaults to []. \n

*@par Outputs:
*y: Reshape tensor with same data as input. The same type as input x. \n

*@par Third-party framework compatibility
*Compatible with the Onnx operator Unsqueeze.
*/

REG_OP(Unsqueeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axes, ListInt, {})
    .OP_END_FACTORY_REG(Unsqueeze)

/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*@li x: A tensor.
*@li axis: The dimension index at which to expand. \n

*@par Outputs:
*y: A tensor with the same data as input, with an additional dimension inserted at the index specified by axis. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ExpandDims.
*/
REG_OP(ExpandDims)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING, DT_BFLOAT16}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(ExpandDims)

/**
*@brief Fills the tensor with the mirror value. \n
*@par Inputs:
* @li x: The tensor to be padded. Format support ND,
* support 1D ~ 5D. Type must be one of the following
* types: int8, uint8, int16, uint16, int32, int64, float16, float,
* double, bool, complex64, complex128, bfloat16.
* @li paddings: A two-column matrix specifying the padding sizes.
* The number of rows has the same rank as "x", type must be int32 or int64. \n

*@par Attributes:
*mode: Either "REFLECT" or "SYMMETRIC". In reflect mode the padded regions
do not include the borders, while in symmetric mode the padded regions
do include the borders. \n

*@par Outputs:
*y: The padded tensor. \n

*@attention Constraints:
*MirrorPad runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MirrorPad.
*/

REG_OP(MirrorPad)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
      DT_INT32, DT_INT64, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL,
      DT_COMPLEX64, DT_COMPLEX128 }))
    .INPUT(paddings, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
      DT_INT32, DT_INT64, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL,
      DT_COMPLEX64, DT_COMPLEX128 }))
    .REQUIRED_ATTR(mode, String)
    .OP_END_FACTORY_REG(MirrorPad)

/**
*@brief Input data for other operators. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*index: Index of the input tensor.The data type must be int32 or int64.
Assume that net has three data nodes, one should be set 0, another should
be set 1, and the left should be set 2. \n

*@par Outputs:
*y: A tensor. \n

*@par Third-party framework compatibility
*Compatible with the Caffe operator Data.
*/
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

/**
*@brief Returns the size of a tensor, that is, an integer of the number of elements of the tensor. \n

*@par Inputs:
*x: A tensor. Must be one of the following types: float32、float16、int8、
int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string. \n

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to "int32". \n

*@par Outputs:
*y: A tensor. The size of the input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Size.
*/
REG_OP(Size)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT32,DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Size)

/**
*@brief Removes dimensions of size 1 from the shape of a tensor. \n

*@par Inputs:
*x: A tensor. All data types are supported. \n

*@par Attributes:
*axis: An optional list of int32 or int64. Defaults to []. If not specified, squeezes all dimensions of size 1.
If specified, only squeezes the dimensions listed. It is an error to squeeze a dimension that is not 1. \n

*@par Outputs:
*y: A tensor. The same type as input x. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Squeeze.
*/
REG_OP(Squeeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(Squeeze)

/**
*@brief Returns the shape of a tensor. \n

*@par Inputs:
*x: A tensor. Must be one of the following types: float32、float16、int8、
int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string、bfloat16. \n

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to int32. \n

*@par Outputs:
*y: A tensor. The shape of the input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Size.
*/
REG_OP(Shape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Shape)

/**
*@brief Creates a constant tensor from a tensor-like object. This operator is used for inference.
Operator Const has the same definition as operator Constant. \n

*@par Attributes:
*value: Required. The value and type of the resulting tensor, and no restrictions on type. \n

*@par Outputs:
*y: A constant tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Const.
*/
REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT4, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)

/**
*@brief Returns an integer representing the rank of input tensor. The rank of a tensor is the number of indices required to uniquely select each element of the tensor, that is, the dimension size of the tensor. \n

*@par Inputs:
*x: A Tensor of type float32, float16, int8, int16, uint16, uint8, int32, int64, uint32, uint64, bool, double, string. \n

*@par Outputs:
*y: A tensor. The rank of input tensor. Type is int32. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Rank.
*/
REG_OP(Rank)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(Rank)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_ARRAY_OPS_H_

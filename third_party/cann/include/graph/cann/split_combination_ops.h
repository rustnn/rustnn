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
 * \file split_combination_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Concatenates tensors along one dimension .

* @par Inputs:
* Two inputs, including:
* @li Dynamic input "x" is A ND Tensor.
* Must be one of the following types: bfloat16, float16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, uint16,
*     complex128, uint32, uint64, qint16, quint16, bool, string.
* @li concat_dim: A 0D Tensor (scalar) with dtype int32, or int64. Specifies the dimension along which to concatenate . \n

* @par Attributes:
* N: An optional int includes all types of int.
* Specifies the number of elements in "x". Defaults to "1". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
* "x" is a list of at least 2 "tensor" objects of the same type . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ConcatV2.
*/
REG_OP(ConcatV2)
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatV2)

/**
* @brief Concatenates tensors along one dimension .

* @par Inputs:
* Two inputs, including:
* @li concat_dim: Must be one of the IndexNumberType: int32, int64.
* Specifies the dimension along which to concatenate .
* @li x: Dynamic input.A ND Tensor.
* Must be one of the BasicType: 
  complex128, complex64, double, float32, float16, int16, int32, int64, int8,
  qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
  bfloat16, complex32, bool. \n


* @par Attributes:
* N: An optional int8, int16, int32, or int64. Specifies the number of elements in "x" .
  Defaults to "1". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
* @li "x" is a list of at least 2 "tensor" objects of the same type.
* @li "concat_dim" is in the range [-len(x.shape), len(x.shape)] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Concat. \n
*/
REG_OP(Concat)
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Concat)
/**
* @brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors .

* @par Inputs:
* Two inputs, including:
* @li split_dim: Must be the following type:int32. Specifies the dimension along which to split.
  Supported format list ["ND"].
* @li x: An ND Tensor.
* Must be one of the types:float16, float32, double, int64, int32, uint8,
  uint16, uint32, uint64, int8, int16, bool, complex64, complex128, qint8,
  quint8, qint16, quint16, qint32, bfloat16.Supported format list ["ND"].

* @par Attributes:
* @li num_split: A required int includes all types of int.
  Specifies the number of output tensors. No default value.

* @par Outputs:
* @li y: Dynamic output.A list of output tensors. Has the same type and format as "x".Supported format list ["ND"].

* @attention Constraints:
* @li "num_split" is greater than or equals to 1.
* @li "num_split" is divisible by the size of dimension "split_dim".
* @li "split_dim" is in the range [-len(x.shape), len(x.shape)-1].

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Split.
*/
REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                          DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                          DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                          DT_BF16,       DT_BOOL}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                                   DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                                   DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                                   DT_BF16,       DT_BOOL}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)

/**
* @brief Splits a tensor along dimension "split_dim" into "num_split"
  smaller tensors according to "size_splits" .

* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor.
* Must be one of the types:float16, float32, double, int64, int32, uint8,
  uint16, uint32, uint64, int8, int16, bool, complex64, complex128, qint8,
  quint8, qint16, quint16, qint32, string, bfloat16.
* @li size_splits: Must be one of the IndexNumberType:int32, int64.
  Specifies a list containing the sizes of each output tensor along the split dimension.
  The elements in "size_splits" sum to the size of dimension "split_dim".
* @li split_dim: Must be the following type:int32, int64. Specifies the
  dimension along which to split. Must be in the range [-len(x.shape), len(x.shape)) . \n

* @par Attributes:
* @li num_split: A required int includes all types of int. Specifies the number of output tensors.
  No default value . \n

* @par Outputs:
* @li y:  Dynamic output.A list of output tensors.
  Has the same type and format as "x" . \n

* @attention Constraints:
* @li Each element in "size_splits" is greater than or equal to 1.
* @li The length of "size_splits" is equal to the value of "num_split".
* @li The elements in "size_splits" sum to the size of dimension "split_dim" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SplitV.
*/
REG_OP(SplitV)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                          DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                          DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                          DT_BF16, DT_BOOL, DT_STRING}))
    .INPUT(size_splits, TensorType::IndexNumberType())
    .INPUT(split_dim, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                                   DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                                   DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                                   DT_BF16, DT_BOOL, DT_STRING}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)

/**
*@brief Packs the list of tensors in values into a tensor with rank one higher
* than each tensor in values, by packing them along the axis dimension.
* Given a list of length N of tensors of shape (A, B, C); if axis == 0 then
* the output tensor will have the shape (N, A, B, C) .

*@par Inputs:
* x: A list of N Tensors. Must be one of the following types: complex128,
* complex64, double, float32, float16, int16, int32, int64, int8, qint16,
* qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8, bfloat16,
* complex32. It's a dynamic input.

*@par Attributes:
*@li axis: An optional int, default value is 0.
*     Dimension along which to pack. The range is [-(R+1), R+1).
*@li N: An optional int, default value is 1. Number of tensors.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Pack.
*/
REG_OP(Pack)
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .ATTR(axis, Int, 0)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Pack)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_

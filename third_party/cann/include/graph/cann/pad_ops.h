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
 * \file pad_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_PAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_PAD_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief Pads a tensor.

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, bfloat16(only support on constant mode),
*     float32, double, int32, uint8, int16, int8, complex64, int64,
*     qint8, quint8, qint32, qint16, quint16, uint16, complex128, uint32, uint64, bool.
* @li paddings: A Tensor of type int32 or int64.
* @li constant_values: A optional Tensor, dtype same as "x"

* @par Attributes:
* @li mode: An optional string, Defaults to "constant", indicates paddings mode,
*     support "constant", "reflect", "edge"
* @li paddings_contiguous: An optional bool value, Defaults to true.
*     If true, paddings is arranged as [[begin0, end0], [begin1, end1], ...]
*     If false, paddings is arranged as [[begin0, begin1], ..., [end0, end1], ...]

* @par Outputs:
* y: A Tensor of the same type as "x".

* @par Third-party framework compatibility:
* Compatible with ONNX operator Pad.
*/
REG_OP(PadV3)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL}))
    .INPUT(paddings, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(constant_values, TensorType::BasicType())
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL}))
    .ATTR(mode, String, "constant")
    .ATTR(paddings_contiguous, Bool, true)
    .OP_END_FACTORY_REG(PadV3)

/**
* @brief Creates a tensor filled with a scalar value.
* This operation creates a tensor of shape "dims" and fills it with "value".
*
* @par Inputs:
* @li dims: A 1D tensor of types int32 or int64. Represents the shape of the output tensor .
        The size of each dimension must be less than or equal to 8. \n

* @li value: A 0D scalar. Specifies the value to fill the returned tensor.
*    Must be one of the following types:
*    bfloat16, float16, float32, double, int32, uint8, int16, int8, complex64, int64, bool,
*    qint8, quint8, qint32, qint16, quint16, uint16, complex128, uint32, uint64, string.
*
* @par Outputs:
* y: A tensor. Has the same type as "value".
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Fill.
* @li Compatible with the Caffe operator Filler.
*
*/
REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_BF16, DT_UINT32, DT_UINT64, DT_STRING}))
    .OP_END_FACTORY_REG(Fill)

/**
* @brief Broadcasts an array for a compatible shape.
*  Broadcasting is the process of making arrays to have compatible shapes
*  for arithmetic operations. Two shapes are compatible if for each
*  dimension pair they are either equal or one of them is one. When trying
*  to broadcast a Tensor to a shape, it starts with the trailing dimensions,
*  and works its way forward.
*
* @par Inputs:
* @li x: A tensor, support all dtype include(BasicType, bool, string, hifloat8, float8_e5m2, float8_e4m3fn).
* @li shape: A tensor.
*     A 1D tensor of type int32 or int64, for the shape of the desired output.
*
* @par Outputs:
* y: A tensor. Has the same tensor info of "x".
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BroadcastTo.
*
*/
REG_OP(BroadcastTo)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(BroadcastTo)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_PAD_OPS_H_

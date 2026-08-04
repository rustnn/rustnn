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
 * \file elewise_calculation_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns x1 * x2 element-wise.
* y = x1 * x2. Support broadcasting operations.

* @par Inputs:
* @li x1: A ND tensor. Must be one of the following types: bool, float16, float32, bfloat16,
* float64, uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.
* @li x2: A ND tensor. Must be one of the following types: bool, float16, float32, bfloat16,
* float64, uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.
* The shape of x1 and x2 must meet the requirements of the broadcast relationship.

* @par Outputs:
* y: A ND tensor. Must be one of the following types: bool, float16, float32, float64, bfloat16,
* uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.

* @attention Constraints:
* "x1" and "x2" have incompatible shapes or types.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Multiply.
*/
REG_OP(Mul)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .OUTPUT(y, "T3")
    .DATATYPE(T1, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                              DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                              DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
    .DATATYPE(T2, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                              DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                              DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Mul)

/**
*@brief Returns x1 + x2 element-wise. Support broadcasting operations.
*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: bool, int8, int16, int32, int64, uint8, float64,
*     float16, bfloat16, float32, complex128, complex64, complex32, string.
* @li x2: A ND Tensor. Must be one of the following types: bool, int8, int16, int32, int64, uint8, float64,
*     float16, bfloat16, float32, complex128, complex64, complex32, string. \n

*@par Outputs:
*y: A ND Tensor. Must be one of the following types: bool, int8, int16, int32, int64, uint8, float64,
*     float16, bfloat16, float32, complex128, complex64, complex32, string.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Add.
*/
REG_OP(Add)
    .INPUT(x1, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .INPUT(x2, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Add)

/**
*@brief Returns x1 - x2 element-wise. Support broadcasting operations.
*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: int8, int16, int32, int64, uint8, float64,
*     float16, float32, complex128, complex64, complex32, uint16, bfloat16, bool.
* @li x2: A ND Tensor of the same dtype as "x1". \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x1".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Subtract.
*/
REG_OP(Sub)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Sub)

/**
*@brief Computes square root of x element-wise.

*@par Inputs:
*  x: A ND Tensor. Must be one of the following types:bfloat16 float16, float32, complex128, complex64, float64. \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Sqrt.
*/
REG_OP(Sqrt)
    .INPUT(x, TensorType{(DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128)})
    .OUTPUT(y, TensorType{(DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128)})
    .OP_END_FACTORY_REG(Sqrt)

/**
* @brief Returns x1/x2 element-wise for real types. Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor.
* Must be one of the following types: bfloat16, float16, float32, double, uint16,
* int8, uint8, int16, int32, int64, complex64, complex128, bool.
* @li x2: A ND Tensor.
* Must be one of the following types: bfloat16, float16, float32, double, uint16,
* int8, uint8, int16, int32, int64, complex64, complex128, bool. \n

* @par Outputs:
* y: A ND Tensor. Has the same dtype and format as input "x1" if the type of "x1" is not bool.
     If the type of "x1" is bool, y is float type. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator RealDiv.
*/
REG_OP(RealDiv)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(RealDiv)

/**
* @brief Returns x1/x2 element-wise for integer types. Support broadcasting operations.

* @par Inputs:
* @li x1: A ND Tensor. Must be one of the following types:
*     float32, float16, bfloat16, int8, uint8, int32, int16,
*     uint16, double, int64, complex64, complex128.
* @li x2: A ND Tensor of the same data type as "x1". \n

* @par Outputs:
* y: A ND Tensor. Has the same dtype as "x1".

* @attention Constraints:
* Broadcasting is supported. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator TruncateDiv. \n

*/
REG_OP(TruncateDiv)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_UINT16, DT_INT16, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_UINT16, DT_INT16, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_UINT16, DT_INT16, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(TruncateDiv)

/**
*@brief Divides "x1/x2" element-wise, rounding toward the
*        most negative integer. Support broadcasting operations.

*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor.
* Must be one of the following types: float16, float32, int32, int64, int8,
*     uint8, int16, uint16, double, bfloat16.
* @li x2: A ND Tensor of the same dtype as "x1". \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x1". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator FloorDiv.
*/
REG_OP(FloorDiv)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8,
                           DT_INT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8,
                           DT_INT64, DT_INT16,DT_UINT16, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8,
                           DT_INT64, DT_INT16,DT_UINT16, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(FloorDiv)

/**
* @brief Computes cosine of "x" element-wise.

* @par Inputs:
* x: A ND Tensor of type bfloat16, float16, float32, double, complex64, complex128.
* the format can be [NCHW,NHWC,ND]

* @par Outputs:
* y: A ND Tensor of the same dtype as "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Cos. \n

*/
REG_OP(Cos)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Cos)

/**
* @brief Computes sine of "x" element-wise.

* @par Inputs:
* One input: \n
* x: An ND Tensor that supports the data type UnaryDataType. \n

* @par Outputs:
* y: An ND Tensor with the same dtype and shape of input "x". \n

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Sin.
*/
REG_OP(Sin)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sin)

/**
* @brief Computes the power of "x1" to "x2". Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types:
*     bfloat16, float16, float32, int32, int64, int8, int16, uint8, double, complex64, complex128.
* @li x2: A ND Tensor of the same dtype as "x1". \n

* @par Outputs:
* y: A ND Tensor. Has the same dtype as "x1". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Pow.
*/
REG_OP(Pow)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .OUTPUT(y, "T3")
    .DATATYPE(T1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_INT8, DT_INT16,
                              DT_UINT8, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .DATATYPE(T2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_INT8, DT_INT16,
                              DT_UINT8, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Pow)

/**
*@brief Cast a tensor form src data type to dst data type.

*@par Inputs:
*One input:
* x:A ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bool, float16, float, int8, int32, uint32, uint8, bfloat16, uint1,
   int64, uint64, int16, uint16, double, complex32, complex64, complex128, qint8, quint8, qint16, quint16, qint32,
   hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2, float4_e2m1.

*@par Attributes:
*dst_type: A required attribute of type int32, specifying the dst data type.

*@par Outputs:
*y:A ND Tensor with same shape as x, and data type is specified by dst_type.

*@attention Constraints:
* @li In the scenario where the data type is converted from float16 to int16: \n
*     If the input data contains inf, inf is converted into the maximum value of int16. \n
*     If the input data contains -inf, -inf is converted into the minimum value of int16. \n
* @li In the scenarios where the data type is converted from INT32 to INT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2048, 1920).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2048, 1920).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT64 and from FLOAT32 to UINT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648, 2147483583).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from INT64 to FLOAT32: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648, 2147483647).
*/
REG_OP(Cast)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                          DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_UINT1,
                          DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                          DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                           DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32,
                           DT_BF16, DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                           DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .REQUIRED_ATTR(dst_type, Int)
    .OP_END_FACTORY_REG(Cast)

/**
* @brief Computes tan of "x" element-wise.

* @par Inputs:
* One input:
* x: A ND Tensor. Must be one of the following types: bfloat16, float16, float32, double,
* complex64, complex128, int32, int64.

* @par Outputs:
* y: An ND or 5HD tensor. Support 1D ~ 8D. A ND Tensor with the same dtype and shape of input "x".

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Tan.
*/
REG_OP(Tan)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Tan)

/**
*@brief Computes the trignometric inverse sine of "x" element-wise.

*
*@par Inputs:
* x: A tensor. Must be one of the following types: float16, bfloat16, float32, float64, int32, int64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same dtype as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Asin.
*
*/
REG_OP(Asin)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE,
                          DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE,
                           DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Asin)

/**
*@brief Computes acos of x element-wise.

*
*@par Inputs:
* x: A tensor. Must be one of the following types: float16, bfloat16, float32,
*     double, int32, int64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same dtype as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Acos.
*
*/
REG_OP(Acos)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Acos)

/**
*@brief Computes inverse hyperbolic cosine of x element-wise.

*
*@par Inputs:
* x: A ND tensor. Dtype must in TensorType::UnaryDataType().
*
*@attention Constraints:
* x Given an input tensor, the function computes inverse hyperbolic cosine of every element.
*   Input range must be [1, inf]. \n
*
*@par Outputs:
* y: A ND tensor. Has the same dtype as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Acosh.
*
*/
REG_OP(Acosh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Acosh)

/**
*@brief Computes inverse hyperbolic sine of x element-wise.
* Given an input tensor, this function computes inverse hyperbolic sine for every element in the tensor.

*
*@par Inputs:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* bfloat16, float16, float32, float64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same dtype as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Asinh.
*
*/
REG_OP(Asinh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Asinh)

/**
*@brief Computes the trignometric inverse tangent of x element-wise.
* The atan operation returns the inverse of tan, such that if y = tan(x) then, x = atan(y).

*
*@par Inputs:
* x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types:
* bfloat16, float16, float32, float64, complex64, complex128.
*
*@par Outputs:
* y: A tensor. Has the same dtype as "x".
* The output of atan will lie within the invertible range of tan, i.e (-pi/2, pi/2).
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Atan.
*
*/
REG_OP(Atan)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Atan)

/**
*@brief Fake-quantize the 'inputs' tensor of type float via global float scalars.

*@par Inputs:
*Three inputs, including:
*@li x: A ND Tensor of type float32. Shape support 1D ~ 8D.
*@li min: A ND Tensor of type float32. Has the same dtype and format as "x".
* Shape must be 1D.
*@li max: A ND Tensor of type float32. Has the same dtype and format as "x".
* [min; max] define the clamping range for the inputs data. Shape must be 1D. \n

*@par Attributes:
*@li num_bits: An optional attribute. Type is int. Defaults to "8".
*@li narrow_range: An optional attribute. Type is bool. Defaults to "False". \n

*@par Outputs:
*y: A ND Tensor of type float32. Shape support 1D ~ 8D. Has the same shape as input "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator FakeQuantWithMinMaxVars.
*/
REG_OP(FakeQuantWithMinMaxVars)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(min, TensorType({DT_FLOAT}))
    .INPUT(max, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxVars)

/**
*@brief Adds 'bias' to 'x'. Support broadcasting operations.

*@par Inputs:
*Two inputs, including:
* @li x: A ND tensor of type NumberType, format list [ND, NCHW, NHWC, NCDHW, NDHWC].
*Must be one of the following types: float32, float64, int32, uint8, int16,
*int8, complex64, int64, qint8, quint8, qint32, bfloat16, uint16, complex128, float16, uint32, uint64.
* @li bias: A 1D tensor with size the C dimension of x:
*when x format is NCHW or NCDHW, C dimension is x.shape[1]. \n
*When x format is NHWC or NDHWC, C dimension is x.shape[-1]. \n
*when x format is ND and data_format is in [NCHW, NCDHW], C dimension is x.shape[1]. \n
*when x format is ND and data_format is in [NHWC, NDHWC], C dimension is x.shape[-1]. \n

*@par Attributes:
*data_format: An optional string. Defaults to "NHWC". \n

*@par Outputs:
*y: A ND tensor with same type and shape and format as "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BiasAdd.
*/
REG_OP(BiasAdd)
    .INPUT(x, TensorType::NumberType())
    .INPUT(bias, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(BiasAdd)

/**
* @brief Compute elementwise modes, such as 0: PRODUCT, 1: SUM, 2: MAX

* @par Inputs:
* One input: An ND or 5HD tensor. Support 1D~8D.
* x: the list of input data, the type of element in Tensor should be same.
*   The max size of x is 32.
*   Should met one of the following types: bfloat16, float16, float32. It's a dynamic input.

* @par Outputs:
* y: A ND Tensor. Has the same dtype and format as "x".

* @par Attributes:
* @li N: A required attribute. the number of input x, max size is 32. Type is int.
* @li model: An optional attribute. Type is int. Defaults to "1".
*    "0": product, "1": sum, "2": max.
* @li coeff: A required attribute. Must met all of following rules:
*    Size of "coeff" must be equal to len("x") or is null.
*    The absolute value of "coeff" must less than or equal to 1. Has the same dtype as "x".
*/
REG_OP(Eltwise)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(N, Int)
    .ATTR(mode, Int, 1)
    .ATTR(coeff, ListFloat, {})
    .OP_END_FACTORY_REG(Eltwise)

/**
*@brief Tests whether the input exceeds a threshold.

*@par Inputs:
* x: A ND Tensor with any format. Must be one of the following types: float16, float32, bfloat16. \n

*@par Attributes:
* threshold: A required float32. Defaults to "0.0". "x" is compared with "threshold", outputs "1" for inputs above threshold; "0" otherwise. \n

*@par Outputs:
* y: A ND Tensor with any format. Has the same dtype as the input. Must be one of the following types: float16, float32, bfloat16.
*@par Third-party framework compatibility
* Compatible with the Caffe operator Threshold.
*/

 REG_OP(Threshold)
     .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
     .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
     .ATTR(threshold, Float, 0.0)
     .OP_END_FACTORY_REG(Threshold);

/**
* @brief Computes the exp(x) - 1 element-wise, y = e^x - 1.

* @par Inputs:
* One input:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* bfloat16, float16, float32, double, complex64, complex128.

* @par Outputs:
* y: A ND Tensor of the same dtype as "x".

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Expm1.
*/
REG_OP(Expm1)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Expm1)

/**
*@brief Returns element-wise smallest integer not less than "x".

*@par Inputs:
* x: A ND Tensor of type bfloat16 or float16 or float32 or float64. \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Ceil.
*/
REG_OP(Ceil)
  .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
  .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
  .OP_END_FACTORY_REG(Ceil)

/**
*@brief Returns element-wise largest integer not greater than "x".

*@par Inputs:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* bfloat16, float16, float32, double.

*@par Outputs:
*y: A ND Tensor of the same dtype as "x".

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator Floor.
*/
REG_OP(Floor)
  .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
  .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
  .OP_END_FACTORY_REG(Floor)

/**
* @brief Computes the logarithm of (x + 1) element-wise, y = ln(x + 1).

* @par Inputs:
* One input:\n
* x: A ND Tensor. Must be one of the following types: bfloat16, float16, float32, double, complex64, complex128. \n

* @par Outputs:
* y: A ND Tensor of the same dtype as "x". \n

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Log1p.
*/
REG_OP(Log1p)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Log1p)

/**
*@brief Returns the truth value of x1 AND x2 element-wise. Support broadcasting operations.

*
*@par Inputs:
*@li x1: A tensor of type bool.
*@li x2: A tensor of the same dtype as "x1".
*
*@attention Constraints:
* LogicalAnd supports broadcasting.
*
*@par Outputs:
* y: A tensor of the same dtype as "x1".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogicalAnd.
*
*/
REG_OP(LogicalAnd)
    .INPUT(x1, TensorType({DT_BOOL}))
    .INPUT(x2, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalAnd)

/**
*@brief Returns the truth value of NOT "x" element-wise.

*@par Inputs:
*x: A ND Tensor of type bool. \n

*@par Outputs:
*y: A ND Tensor of type bool. \n

*@attention Constraints:
* The input and output values are "1" or "0", corresponding to bool values "true" and "false". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator logical_not.
*/
REG_OP(LogicalNot)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalNot)

/**
*@brief Returns the max of "x1" and "x2" (i.e. x1 > x2 ? x1: x2) element-wise. Support broadcasting operations. \n

*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: float16, float32, double, int32, int64, bfloat16, int8, uint8.
* @li x2: A ND Tensor of the same dtype as "x1". \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x1". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Maximum.
*/
REG_OP(Maximum)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_BF16, DT_INT8, DT_UINT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_BF16, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_BF16, DT_INT8, DT_UINT8}))
    .OP_END_FACTORY_REG(Maximum)

/**
* @brief Returns the min of "x1" and "x2" (i.e. x1 < x2 ? x1: x2) element-wise. Support broadcasting operations. \n

* @par Inputs:
* Two inputs, include:
* @li x1: A ND Tensor. Must be one of the following types: bfloat16, float32, float16, double, int32, int64, int8, uint8.
* @li x2: A ND Tensor of the same dtype as "x1". \n

* @par Outputs:
* y: A ND Tensor of the same dtype as "x1". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator Minimum.
*/
REG_OP(Minimum)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
    .OP_END_FACTORY_REG(Minimum)

/**
*@brief Returns the truth value of (x = y) element-wise. Support broadcasting operations. \n

*@par Inputs:
* Two inputs, including:
*@li x1: A ND Tensor. Must be one of the following types:
*    bfloat16, float16, float32, int32, int8, uint8, double, int16, int64, complex64,
*    complex128, quint8, qint8, qint32, string, bool. the format can be [NCHW, NHWC, ND]
*@li x2: A ND Tensor of the same dtype and format as "x1". \n

*@par Outputs:
*y: A ND Tensor. Has the bool dtype. True means x1 == x2, false means x1 != x2.

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Equal.
*/
REG_OP(Equal)
    .INPUT(x1, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_INT16, DT_INT64, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QUINT8, DT_QINT8, DT_QINT32,
                           DT_STRING, DT_BOOL}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_INT16, DT_INT64, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QUINT8, DT_QINT8, DT_QINT32,
                           DT_STRING, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Equal)

/**
*@brief Computes the reciprocal of "x".

*@par Inputs:
*One inputs, include:
*x:A ND Tensor of type float16, float32, double,
*     complex64, complex128, bfloat16. the format can be [NCHW,NHWC,ND]

*@par Outputs:
*y:A ND Tensor with same type as "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Reciprocal.
*/
REG_OP(Reciprocal)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16,
                          DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .OP_END_FACTORY_REG(Reciprocal)

/**
*@brief Computes square of "x" element-wise.

*@par Inputs:
*One input:
* x: A ND Tensor. Must be one of the following types: float16, bfloat16, float32, float64, int32, int64, complex64,
*    complex128.

*@par Outputs:
*y: An ND or 5HD tensor. Support 1D ~ 8D. Shape and dtype of output, should be same shape and type as input.

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Square.
*/
REG_OP(Square)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BF16,
                          DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_BF16,
                           DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Square)

/**
*@brief Computes the sign  of "x". \n

*@par Inputs:
* x:An ND Tensor of type bfloat16, float16, float32, int32, int64, double,
*     complex64, complex128. \n

*@par Outputs:
*y:An ND Tensor with same type as "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Sign.
*/
REG_OP(Sign)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                          DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Sign)

/**
* @brief Computes the exponential of "x" element-wise.

* @par Inputs:
* One input:
* x: A ND tensor. Must be one of the following types: bfloat16, float16, float32, double, complex64, complex128.
* Only when x's dtype is bfloat16, float16 or float32, attributes are valid and can be set.
* When x's dtype is double, complex64, complex128, attributes are invalid.

* @par Attributes:
* @li base: An optional attribute of type float32, specifying the base gamma. Must be positive or "-1.0", defaults to "-1.0".
* @li scale: An optional attribute of type float32, specifying the scale alpha. Defaults to "1.0".
* @li shift: An optional attribute of type float32, specifying the shift beta. Defaults to "0.0".

* @par Outputs:
* y: A ND tensor of the same dtype as "x".

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Exp.
*/
REG_OP(Exp)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .ATTR(base, Float, -1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Exp)

/**
* @brief Returns element-wise remainder of division.
* Consistent with: floor(x1/x2) * x2 + mod(x1, x2) = x1.
* Integer division by zero on NPU returns x1. Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND tensor. Must be one of the following types:
*    int32, int64, float, float16, double, bfloat16
* @li x2: A ND tensor. Must have the same dtype as "x1".
*
* @par Outputs:
* y: A ND tensor. Has the same dtype as "x1".

* @attention Constraints:
* @li x2: The input data does not support 0
* @li When value of tensor exceeds 2048 , the accuracy of operator cannot guarantee the
* requirement of double thousandths in the mini platform
* @li Due to different architectures, the calculation results of this operator
* on NPU and CPU may be inconsistent
* @li If shape is expressed as (D1,D2... ,Dn), then D1*D2... *DN<=1000000,n<=8

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator FloorMod.
*/
REG_OP(FloorMod)
    .INPUT(x1, TensorType({DT_INT32, DT_INT64, DT_FLOAT,
                           DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .INPUT(x2, TensorType({DT_INT32, DT_INT64, DT_FLOAT,
                           DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64, DT_FLOAT,
                           DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(FloorMod)

/**
*@brief Returns the truth value of (x1 >= x2) element-wise. Support broadcasting operations. \n
*When input is int32 and (x2 - x1) > 2^31 or < -2^31,
*aicore accuracy is not guaranteed.

*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor with TensorType::RealNumberType().
* @li x2: A ND Tensor to be compared to "x1", and the data type is the same as "x1".

*@par Outputs:
*y: A ND Tensor. Has the bool dtype. True means x1 >= x2, false means x1 < x2.

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator GreaterEqual.
*/
REG_OP(GreaterEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(GreaterEqual)

/**
*@brief Returns the truth value of (x1 > x2) element-wise. Support broadcasting operations. \n
*When input is int32 and (x2 - x1) > 2^31 or < -2^31,
*aicore accuracy is not guaranteed.

*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor with TensorType::RealNumberType().
* @li x2: A ND Tensor to be compared to "x1", and the data type is the same as "x1".

*@par Outputs:
*y: A ND Tensor. Has the bool dtype. True means x1 > x2, false means x1 <= x2.

* @attention Constraints:
* Broadcasting is supported. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Greater. \n

*/
REG_OP(Greater)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Greater)

/**
*@brief Returns the truth value of (x1 < x2) element-wise. Support broadcasting operations. \n
*When input is int32 and (x2 - x1) > 2^31 or < -2^31,
*aicore accuracy is not guaranteed.

*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor with TensorType::RealNumberType().
* @li x2: A ND Tensor to be compared to "x1", and the data type is the same as "x1".

*@par Outputs:
*y: A ND tensor. Has the bool dtype. True means x1 < x2, false means x1 >= x2.

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator Less.
*/
REG_OP(Less)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Less)

/**
* @brief Return element-wise integer closest to x.

* @par Inputs:
* One input, include:
* x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types:
* float16, float32, double, bfloat16.
*
* @par Outputs:
* y: A mutable Tensor. Has the same dtype as "x".
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Rint.
*/
REG_OP(Rint)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(Rint)

/**
*@brief Rounds the values of a tensor to the nearest integer, element-wise.
 * Rounds half to even.

*@par Inputs:
*Inputs including:
* x: An ND Tensor of type bfloat16, float16, float, int64, double, int32.

* @par Attributes:
* decimals: An optional int attr, number of decimal places to round to. Defaults to "0".

*@par Outputs:
*y: An ND Tensor. Has the same data type and shape as "x".
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Round.
*@attention Constraints:
* @li When the input value is between [-0.5, -0], the output value is 0.
* @li In the scenarios where the decimals is not zero:
*     The input data exceeds the range of (-347000, 347000), which may affect the precision errors.
*/
REG_OP(Round)
    .INPUT(x, TensorType(DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32, DT_INT64,
                         DT_DOUBLE))
    .OUTPUT(y, TensorType(DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32, DT_INT64,
                          DT_DOUBLE))
    .ATTR(decimals, Int, 0)
    .OP_END_FACTORY_REG(Round)

/**
*@brief Computes reciprocal of square root of "x" element-wise: y = 1/sqrt{x}.

*
*@par Inputs:
* x: An ND or 5HD tensor. Must be one of the following types: bfloat16, float, double, float16,
 * complex64, complex128.
*
*@par Outputs:
* y: An ND or 5HD tensor. Has the same dtype as "x".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Rsqrt.
*
*/
REG_OP(Rsqrt)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Rsqrt)

/**
* @brief Computes logarithm of x element-wise.
* y = log_base(shift + scale * x), with "base" > 0.

* @par Inputs:
* x: A ND Tensor of type uint8, int8, int16, int32, int64, float64,
*    float16, bfloat16, float32, bool, complex128 or complex64. \n

* @par Attributes:
* @li base: An optional float32, specifying the base "e". Defaults to "-1.0"

* @li scale: An optional float32, specifying the scale of input "x". Defaults
* to "1.0"
* @li shift: An optional float32, specifying the shift. Defaults to "0.0"

* @par Outputs:
* y: A tensor, when the input is of integer type, the y type is float32.
*    Other case, y has same type as "x". \n

* @attention Constraints:
* @li "base" is supposed to be greater than 0. Retaining the default
* value "-1" sets "base" to "e".
* @li If the input value of operator Log is within the range (0, 0.01] or
* [0.95, 1.05], the output accuracy is subject to change. \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Log.
* @li Compatible with the Caffe operator Log.
*/
REG_OP(Log)
    .INPUT(x, TensorType({DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BF16,
                          DT_BOOL, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType::UnaryDataType())
    .ATTR(base, Float, -1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Log)

/**
*@brief Returns the truth value of x1 OR x2 element-wise. Support broadcasting operations.

*
*@par Inputs:
*@li x1: A ND tensor of type bool.
*@li x2: A ND tensor of the same dtype as "x1".
*
*@attention Constraints:
* LogicalOr supports broadcasting.
*
*@par Outputs:
* y: A tensor of the same dtype as "x1".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogicalOr.
*
*/
REG_OP(LogicalOr)
    .INPUT(x1, TensorType({DT_BOOL}))
    .INPUT(x2, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalOr)

/**
*@brief Computes numerical negative value element-wise (y = -x)

*@par Inputs:
* One input:
*x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* float16, float32, int32, int64, complex64, complex128, bfloat16, int8, float64.

*@par Outputs:
*y: A ND Tensor. Has the same dtype and format as input "x".

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Neg.
*/
REG_OP(Neg)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_INT8, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_INT8, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Neg)

/**
*@brief Returns the index with the largest value across axes of a tensor.

*@par Inputs:
* Two inputs, including:
*@li x: A ND and multi-dimensional tensor of type float16, float32, int32, or int16.
*@li dimension: A Scalar of type int32, specifying the index with the largest value.

*@par Attributes:
*dtype: The output type, either "int32" or "int64". Defaults to "int64".

*@par Outputs:
*y: A ND tensor of type int32 or int64, specifying the index with the largest value. The dimension is one less than that of "x".

*@attention Constraints:
*@li x: If there are multiple maximum values, the index of the first maximum value is used.
*@li The value range of "dimension" is [-dims, dims - 1]. "dims" is the dimension length of "x".

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ArgMax.
*/
REG_OP(ArgMaxV2)
    .INPUT(x, TensorType::NumberType())
    .INPUT(dimension, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(ArgMaxV2)

/**
* @brief Returns the truth value of (x1 != x2) element-wise. Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor with TensorType::RealNumberType().
* @li x2: A ND Tensor to be compared to "x1", and the data type is the same as "x1".

*@par Outputs:
*y: A ND Tensor. Has the bool dtype. True means x1 != x2, false means x1 == x2.

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator NotEqual.
*/
REG_OP(NotEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(NotEqual)

/**
*@brief Returns the truth value of (x1 <= x2) element-wise. Support broadcasting operations. \n
*When input is int32 and (x2 - x1) > 2^31 or < -2^31,
*aicore accuracy is not guaranteed.

*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor with TensorType::RealNumberType().
* @li x2: A ND Tensor to be compared to "x1", and the data type is the same as "x1".

*@par Outputs:
*y: A ND Tensor. Has the bool dtype. True means x1 <= x2, false means x1 > x2.

*@par Third-party framework compatibility:
* Compatible with the TensorFlow operator LessEqual.
*/
REG_OP(LessEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LessEqual)

/**
*@brief Returns (x1 - x2)(x1 - x2) element-wise. Support broadcasting operations.

*@par Inputs:
*Two inputs, including: \n
*@li x1: A ND Tensor. Must be one of the following types: bfloat16, float16, float32,
* float64, int32, int64, complex64, complex128.
*@li x2: A ND Tensor. Has the same dtype as "x1".
* The shape of x1 and x2 must meet the requirements of the broadcast relationship. \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x1". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator SquaredDifference.
*/
REG_OP(SquaredDifference)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32,
                           DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(SquaredDifference)

/**
* @brief Clips tensor values to a specified min and max.
* When the input is bfloat16, float16, float32, int32 or int64, broadcasting operations are supported.  \n

* @par Inputs:
* Three inputs, including:
* @li x: A ND tensor of type complex128, complex64, double, float32, float16, int16，
* int32, int64, int8, qint32, qint8, quint8, uint16, uint8, bfloat16, complex32.
* @li clip_value_min: A ND tensor of the same dtype as "x".
* @li clip_value_max: A ND tensor of the same dtype as "x". \n

* @par Outputs:
* y: A ND tensor. Has the same dtype as "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ClipByValue.
*/
REG_OP(ClipByValue)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, 
                          DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_QINT32, DT_QINT8,
                          DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .INPUT(clip_value_min, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, 
                                       DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_QINT32, DT_QINT8,
                                       DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .INPUT(clip_value_max, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, 
                                       DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_QINT32, DT_QINT8,
                                       DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, 
                            DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_QINT32, DT_QINT8,
                            DT_QUINT8, DT_UINT16, DT_UINT8, DT_BF16, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(ClipByValue)

/**
* @brief Clips tensor values to a specified min and max.
* When the input is bfloat16, float16, float32, int32 or int64, broadcasting operations are supported.  \n

* @par Inputs:
* Three inputs, including:
* @li x: A ND tensor with TensorType::NumberType().
* @li clip_value_min: A ND tensor of the same dtype as "x".
* @li clip_value_max: A ND tensor of the same dtype as "x". \n

* @par Outputs:
* y: A ND tensor. Has the same dtype as "x". \n

* @par Third-party framework compatibility
* Compatible with the PyTorch operator clip.
*/
REG_OP(ClipByValueV2)
    .INPUT(x, TensorType::NumberType())
    .INPUT(clip_value_min, TensorType::NumberType())
    .INPUT(clip_value_max, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(ClipByValueV2)

/**
*@brief Computes y = x1 * log(x2). Support broadcasting operations.

*@par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: bfloat16, float16, float32,
* double, complex64, complex128.
* @li x2: A ND Tensor. Has the same dtype as "x1". \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x1". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator Xlogy.
*/
REG_OP(Xlogy)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Xlogy)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_ELEWISE_CALCULATION_OPS_H_

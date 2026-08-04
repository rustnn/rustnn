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
 * \file nonlinear_fuc_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Compute sigmoid of "x" element-wise .

* @par Inputs:
* A Tensor of type complex64, complex128, bfloat16, float16, float32 or double . \n

* @par Outputs:
* A Tensor. Has the same type as "x" . \n

* @see Relu()

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Sigmoid.
*/
REG_OP(Sigmoid)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sigmoid)

/**
* @brief Computes rectified linear: "max(x, 0)".
*
* @par Inputs:
* x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types:
* float32, float64, int32, uint8, int16, int8, int64, uint16, float16, qint8, bfloat16.
*
* @par Outputs:
* y: A tensor. Has the same type as "x".
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Relu.
* @li Compatible with the Caffe operator ReLULayer.
*
*/
REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                          DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_QINT8, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                           DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_QINT8, DT_BF16}))
    .OP_END_FACTORY_REG(Relu)

/**
* @brief Computes rectified linear 6.
* activations = min(max(x, 0), 6) .

* @par Inputs:
* x: A ND Tensor of type RealNumberType(includes: double, float32, float16,
* int16, int32, int64, int8, uint16, uint32, uint64, uint8, bfloat16) . \n

* @par Outputs:
* y: A ND Tensor with the same type as x . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Relu6.
*/
REG_OP(Relu6)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(Relu6)

/**
 * @brief ThresholdedRelu takes one input data (Tensor) and produces one output data (Tensor)
 *  where the rectified linear function, y = x for x > alpha, y = 0 otherwise, is applied to the tensor elementwise.
 *
 * @par Inputs:
 * one input including:
 * x: input A Tensor. Must be one of the following types: float32, float16
 *
 * @par Attributes:
 * alpha: An optional float. Defaults to 1.0. \n

 * @par Outputs:
 * one output including:
 * y:A Tensor of the same type as x
 *
 */
REG_OP(ThresholdedRelu)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(alpha, Float, 1.0)
    .OP_END_FACTORY_REG(ThresholdedRelu)

/**
* @brief Compute hard_swish of "x" element-wise .

*@par Inputs:
*One input, including:
*x: A Tensor. Must be one of the following types: float16, float32, bfloat16

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*@par Third-party framework compatibility
* Compatible with the Torch operator HardSwish.
*/
REG_OP(HardSwish)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(HardSwish)

/**
* @brief Thresholds each element of the input Tensor: y = (x > threshold) ? x : value

* @par Inputs:
* Three inputs, including:
* @li x: A ND Tensor. Support 1D~8D.
* Must be one of the following types: float16, float32, int8, int32, uint8, int64, bfloat16. \n
* @li threshold: A Tensor which should have the shape (1,), the value to threshold at.
* Must be one of the following types: float16, float32, int8, int32, uint8, int64, bfloat16. \n
* @li value: A Tensor which should have the shape (1,), the value to replace with. default value is 0.
* Must be one of the following types: float16, float32, int8, int32, uint8, int64, bfloat16. \n

* @par Outputs:
* y: A Tensor which has the same shape, format and type as the input x. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Threshold.
*/
REG_OP(ThresholdV2)
     .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_BF16}))
     .INPUT(threshold, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_BF16}))
     .OPTIONAL_INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_BF16}))
     .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_BF16}))
     .OP_END_FACTORY_REG(ThresholdV2)

/**
* @brief Performs parametric ReLU .

* @par Inputs:
* Two inputs, including:
* @li x: A multi-dimensional Tensor of type bfloat16, float16 or float32.
* @li weight: A Scalar or 1D Tensor of type bfloat16, float16 or float32, specifying the weight,
* initial value of "a". The number of dimensions must be the same as the number of channels . \n

* @par Outputs:
* y: An activated Tensor. Has the same dimensions with "x" . \n

* @par Third-party framework compatibility
* Compatible with PyTorch and Caffe operator PReLU.
*/
REG_OP(PRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(PRelu)

/**
*@brief Computes the for the Swish of "x" .

*@par Inputs:
*One input, including:
* x: A tensor, which supports 1D-8D defaultly and must be one of the following types: float16, bfloat16, float32. \n

*@par Outputs:
* y: A tensor of the same type, shape and format as "x", and y = x / (1 + e ^ (-scale * x)). \n

*@par Attributes:
* scale: scalar parameter, the multiplier of x. Must be one of the following types: float. Default value = 1.0. \n

*@par Third-party framework compatibility
*Compatible with the Torch operator Swish
*/
REG_OP(Swish)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(scale, Float, 1.0)
    .OP_END_FACTORY_REG(Swish)

/**
*@brief Computes hyperbolic tangent of "x" element-wise .

*@par Inputs:
* One input:
* x: An ND tensor. support 1D ~ 8D. Must be one of the following types:
* float16, float32, bfloat16.
*
*@par Outputs:
* y: A Tensor. Has the same type as "x" .
*
*@par Third-party framework compatibility
* Compatible with TensorFlow operator Mish.
*/

REG_OP(Mish)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OP_END_FACTORY_REG(Mish)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_

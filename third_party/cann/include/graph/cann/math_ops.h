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
 * \file math_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Computes the output as (shift + scale * x) ^ power .

* @par Inputs:
* x: A tensor of type float16, float32 or bfloat16 . \n

* @par Attributes:
* @li power: Optional. Must be one of the following types: float32. Defaults to 1.0.
* @li scale: Optional. Must be one of the following types: float32. Defaults to 1.0.
* @li shift: Optional. Must be one of the following types: float32. Defaults to 0.0 . \n

* @par Outputs:
* y: A tensor. Has the same type and shape as "x".
* @par Third-party framework compatibility
* Compatible with the Caffe operator Power.
*/

REG_OP(Power)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(power, Float, 1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Power);

/**
* @brief Computes the Gauss error function of `x` element-wise . \n

* @par Inputs:
* x: A Tensor of type bfloat16, float16, float32 or double. the format can be
*    [NCHW,NHWC,ND]

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Erf.
*/
REG_OP(Erf)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Erf)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_

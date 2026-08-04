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
 * \file nn_batch_norm_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_

#include "graph/operator_reg.h"

namespace ge {


/**
* @brief Performs batch normalization .

* @par Inputs:
* @li x: A 4D or 5D Tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
* @li mean: A 1D Tensor of type float32 or float16 or bfloat16, the shape is same as dim C of input x.
* Specifies the mean used for inference.
* @li variance: A 1D Tensor of type float32 or float16 or bfloat16, the shape is same as dim C of input x.
* Specifies the variance used for inference.
* @li momentum: A 1D Tensor of type float32 or float16 or bfloat16, the shape is same as dim C of input x.
* represents the mean and the variance's scale factor
* @li scale: An optional 1D tensor of type float16 or float32 or bfloat16, the shape is same as dim C of input x.
* @li offset: An optional 1D tensor of type float16 or float32 or bfloat16, the shape is same as dim C of input x.
* @par Attributes:
* @li epsilon: An optional float32, specifying the small value added to variance to avoid dividing by zero.
      Defaults to "0.00001".
* @li use_global_stats: An optional bool, mean inference mode, only can be "True".
* @li mode: An optional int, defaults to "1".
* @par Outputs:
* @li y: A 4D or 5D Tensor of type float16 or float32 or bfloat16 for the normalized "x"
*/
REG_OP(BNInference)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(momentum, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float,1e-5f)
    .ATTR(use_global_stats, Bool,true)
    .ATTR(mode, Int,1)
    .OP_END_FACTORY_REG(BNInference)

/**
*@brief Normalizes elements of a specific dimension of eigenvalues (L2) .

*@par Inputs:
*x: A ND Tensor(1D-8D) of type float16 or float32, specifying the eigenvalue . \n

*@par Attributes:
*@li axis: A optional required attribute of type list, specifying the axis for normalization Defaults to {} .
*@li eps: An optional attribute of type float, specifying the lower limit of normalization. Defaults to "1e-4" . \n

*@par Outputs:
*y: A ND Tensor(1D-8D) of type float16 or float32, specifying the eigenvalue for normalization. \n

*@par Third-party framework compatibility
* Compatible with the L2 scenario of PyTorch operator Normalize.
*/
REG_OP(L2Normalize)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {})
    .ATTR(eps, Float, 1e-4f)
    .OP_END_FACTORY_REG(L2Normalize)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_BATCH_NORM_OPS_H_

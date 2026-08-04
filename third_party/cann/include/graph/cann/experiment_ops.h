/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2025. All rights reserved.
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
 * \file experiment_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief Finds values and indices of the "k" largest elements for the last
* dimension . \n

* @par Inputs:
* Two inputs, including:
* @li x: A 1D-8D tensor, with the last dimension at least "k".
* Supported type: float16, float32. Supported format: ND.
* @li k: A 0D Tensor of type int32. Supported format: ND.
* Number of top elements to look for along the last dimension (along each row
* for matrices) . \n

* @par Attributes:
* @li sorted: An optional bool. Defaults to "True".
* If "True", the returned "k" elements are themselves sorted.
* If "False", the returned "k" elements are not sorted.
* @li dim: An optional int. Defaults to "-1". For reserved use.
* @li largest: An optional bool, controls whether to return largest or smallest elements. Defaults to "True".
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
REG_OP(TopKV3)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .ATTR(sorted, Bool, true)
    .ATTR(dim, Int, -1)
    .ATTR(largest, Bool, true)
    .OP_END_FACTORY_REG(TopKV3)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_

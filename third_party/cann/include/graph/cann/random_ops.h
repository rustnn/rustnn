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
 * \file random_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_RANDOM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_RANDOM_OPS_H_

#include <vector>

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Permutes data in the channel dimension of the input

*@par Inputs:
*Inputs including:
* x: A required Tensor. Must be one of the following types:
     float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64 . \n

*@par Attributes:
* group: A required int32, specifying the number of groups to split the channel dimension into. Defaults to "1" . \n

*@par Outputs:
* y: A required Tensor. Has same type and shape as "x". Must be one of the following types:
     float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64 . \n

*@attention Constraints:
*@li "group" must be greater than 0 and must evenly divide the channel dimension size.
*@li The format of input "x" must be NCHW.
*@par Third-party framework compatibility
* Compatible with the Caffe operator ShuffleChannel.
*/
REG_OP(ShuffleChannel)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16,
                          DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT8, DT_UINT8, DT_INT16,
                           DT_UINT16, DT_INT32, DT_UINT32,DT_INT64,DT_UINT64}))
    .ATTR(group, Int, 1)
    .OP_END_FACTORY_REG(ShuffleChannel)
}   // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_RANDOM_OPS_H_

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
 * \file sparse_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Converts a sparse representation into a dense tensor. \n

*@par Inputs:
* @li indices: A 0D, 1D, or 2D Tensor of type int32 or int64.
* @li output_shape: A 1D Tensor of the same type as "indices". The shape of the dense output tensor.
* @li values: A 1D Tensor. Values corresponding to each row of "indices",
or a scalar value to be used for all sparse indices.
* @li default_value: A Tensor of the same type as "values". \n

*@par Attributes:
*validate_indices: If true, indices are checked to make sure they are sorted in
lexicographic order and that there are no repeats. \n

*@par Outputs:
*y: A Tensor. Has the same type as "values". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseToDense.
*/
REG_OP(SparseToDense)
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(output_shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_DOUBLE}))
    .INPUT(default_value, TensorType({DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL, \
        DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_DOUBLE}))
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(SparseToDense)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_
/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef _CCE_GRAPH_OP_GE_OP_DETECTION_DEFS_H
#define _CCE_GRAPH_OP_GE_OP_DETECTION_DEFS_H

#include "graph/compatible/operator_reg.h"
#include "graph/op/detection_defs.h"

namespace ge {
/*
 * Permutes the dimensions of the input according to a given pattern.
 * <Input>
 *    x : Input tensor, of the same type as Data.
 *    w : Deprecated, will be removed in a future version.
 * <Output>
 *    y : Has the same shape as the input, but with the dimensions re-ordered according to the specified pattern.
 * <Attr>
 *    order : Tuple of dimension indices indicating the permutation pattern, whose size >= dimensions of 'x'.
 *    For instance, x = (4, 4, 4), order = (0, 2, 1).
 * <Added in HiAI version>
 *    100.300.010.011
 */

CPT_REG_OP(Permute)
.CPT_INPUT(x, TensorType ({ DT_FLOAT }))
.CPT_OPTIONAL_INPUT(w, TensorType ({ DT_FLOAT }))
.CPT_OUTPUT(y, TensorType ({ DT_FLOAT }))
.CPT_ATTR(order, AttrValue::LIST_INT { 0 })
.CPT_OP_END()
}  // namespace ge

#endif  // _CCE_GRAPH_OP_GE_OP_DETECTION_DEFS_H

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

#ifndef _CCE_GRAPH_OP_GE_OP_CONST_DEFS_H
#define _CCE_GRAPH_OP_GE_OP_CONST_DEFS_H

#include "graph/compatible/operator_reg.h"

namespace ge {
/*
 * Constant tensor
 * <Output>
 *    y : Output tensor containing the same value of the provided tensor
 * <Attr>
 *    value : Value for the elements of the output tensor
 * <Added in HiAI version>
 *    100.300.010.011
 */
CPT_REG_OP(Const)
.CPT_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT8, DT_INT32, DT_BOOL }))
.CPT_ATTR(value, AttrValue::TENSOR(new (std::nothrow) Tensor(TensorDesc())))
.CPT_OP_END()
} // namespace ge

#endif // _CCE_GRAPH_OP_GE_OP_CONST_DEFS_H
